// ego_velocity.ino — ESP32 + Adafruit BNO08x IMU: measures the platform's own planar motion —
// forward velocity, lateral (sideways) velocity, AND yaw rate (angular velocity) — and sends all
// three to a Raspberry Pi over a dedicated hardware UART (GPIO pins, not the ESP32's USB port),
// for the radar-fusion pipeline's ego-motion compensation. See EGO_VELOCITY.md in the repo root
// for the full design, wiring diagram, and why this approach was chosen.
//
// Yaw rate matters here specifically because the radar cannot measure it at all — a single radar
// point only ever gives a radial (range-rate) measurement, so there is no way to recover the
// platform's own rotation rate from radar data alone. The gyroscope is the only sensor in this
// project that can supply it.
//
// Builds on the original gyro-test.ino (I2C setup, BNO08x init at 0x4B, INT/RESET pins) — same
// board pinout, extended with:
//   - SH2_LINEAR_ACCELERATION (gravity already removed by the BNO08x's own sensor fusion) and
//     SH2_GYROSCOPE_CALIBRATED, instead of just the rotation vector quaternion
//   - integrating linear acceleration into a 2-axis (forward + lateral) velocity estimate, with
//     two forms of drift correction: a startup bias calibration, and a zero-velocity update (ZUPT)
//     that pins the estimate back to 0 whenever the platform is detected to be genuinely stationary
//   - yaw rate is sent as a DIRECT gyro reading, not integrated — so unlike the velocity estimate
//     it carries no dead-reckoning drift of its own (see "IMPORTANT LIMITATION" below for what that
//     does and doesn't cover)
//   - sending the result out a SEPARATE hardware UART (Serial2) wired directly to the Pi's GPIO
//     UART pins, keeping the native Serial/USB port free for flashing and debug prints
//
// ---------------------------------------------------------------------------------------------
// IMPORTANT LIMITATION — read before trusting this for anything safety-critical
//
// Forward/lateral velocity is dead-reckoning: velocity = integral of acceleration over time, with
// no independent ground-truth correction (no wheel encoder, no GPS, no vision odometry). Any
// residual accelerometer bias integrates into an ever-growing velocity error. The ZUPT logic below
// bounds that error TO ZERO every time the platform genuinely stops — which is why it matters —
// but between stops, on a long uninterrupted drive, expect real drift. This is good enough to
// correct the radar's Doppler ego-compensation (a few tenths of a m/s of error there just nudges a
// static/moving point classification, not a hard safety threshold) — it is NOT a substitute for a
// wheel encoder or GPS if you need an accurate absolute speed over a long, continuous drive.
//
// Yaw RATE (this file) does not have that problem — it's a direct, bias-calibrated gyro reading,
// not an integral, so it doesn't drift the way velocity does. But note this file does NOT integrate
// yaw rate into a heading/orientation angle — only the instantaneous rate is sent. Turning that
// into "how far has the platform rotated since frame N" (needed to fully correct track *positions*,
// not just velocities, during a turn) would require integrating this rate over time and is a
// further step this firmware does not attempt.
//
// ---------------------------------------------------------------------------------------------
// MOUNTING REQUIREMENT
//
// This firmware assumes the BNO08x is rigidly mounted with its own +X axis pointing in the
// platform's direction of travel (forward) and +Y axis pointing right — matching this project's
// existing radar convention (x = right, y = forward; see iwr1642_live.py). It does NOT use the
// rotation-vector quaternion to figure out orientation dynamically — a deliberate simplification
// for this version (see EGO_VELOCITY.md for the more robust quaternion-based version, left as a
// documented future improvement).
//
// If your specific breakout's silkscreen doesn't make X/Y obvious, or you can't mount it exactly
// this way: push the board by hand in the forward direction and confirm the sign of ax in the
// Serial debug output (should go positive/negative consistently with "forward"), then do the same
// pushing right for ay — and flip FORWARD_SIGN / RIGHT_SIGN below if either comes out backwards,
// rather than trying to physically remount the board.

#include <Wire.h>
#include <Adafruit_BNO08x.h>

// -------------------------------------------------------------------------------- IMU (I2C)
#define BNO08X_INT   4
#define BNO08X_RESET 15
#define BNO08X_I2C_ADDR 0x4B

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// -------------------------------------------------------------------------------- Pi UART link
// A SEPARATE hardware UART from the native USB Serial, so flashing/debugging over USB never
// conflicts with the data link to the Pi. GPIO16/17 are the classic ESP32 UART2 default pins on
// most dev boards — change these two macros if your specific board maps them elsewhere.
#define PI_UART_RX_PIN 16   // ESP32 RX2  <- Pi TX  (Pi header pin 8,  GPIO14)
#define PI_UART_TX_PIN 17   // ESP32 TX2  -> Pi RX  (Pi header pin 10, GPIO15)
#define PI_UART_BAUD   115200
// Both the ESP32 and the Pi run 3.3V logic — connect TX/RX/GND directly, NO level shifter needed
// (unlike a 5V Arduino Uno, which would need one). Do not skip the common GND connection.

// -------------------------------------------------------------------------------- axis mapping
// See the MOUNTING REQUIREMENT note above — adjust these two after physically confirming your
// board's actual orientation. A wrong sign here silently flips "closing" vs "receding" (FORWARD_SIGN)
// or "left" vs "right" (RIGHT_SIGN) in every downstream ego-compensated value.
#define FORWARD_SIGN 1.0f   // BNO08x +X assumed forward
#define RIGHT_SIGN   1.0f   // BNO08x +Y assumed right

// -------------------------------------------------------------------------------- tuning
// Startup bias calibration: average this many samples (assumes the platform is stationary at
// power-on — if it isn't, the estimate will be off by whatever it was actually doing right then).
const int CAL_SAMPLES = 100;                  // ~2 s at the 50 Hz report rate below

// Zero-velocity update (ZUPT): the actual drift-bounding mechanism for forward/lateral velocity
// (see IMPORTANT LIMITATION above — this does not apply to yaw rate, which isn't integrated). If
// the bias-corrected forward AND lateral acceleration, AND the yaw rate, all stay under these
// thresholds for ZUPT_HOLD_MS continuously, the platform is declared stationary and both velocity
// components are pinned to 0 rather than integrated — otherwise a small residual bias would make
// "speed" climb forever while actually parked.
const float ACCEL_ZUPT_THRESH_MPS2 = 0.06f;   // m/s^2 — BNO08x linear-accel noise floor is roughly here
const float GYRO_ZUPT_THRESH_DPS   = 1.0f;    // deg/s — treat a slow turn as "not stationary" too
const unsigned long ZUPT_HOLD_MS   = 300;     // must stay under both thresholds this long before zeroing

// -------------------------------------------------------------------------------- state
float bias_ax = 0.0f, bias_ay = 0.0f;
int cal_count = 0;
bool calibrated = false;

float vy_mps = 0.0f;          // forward velocity (integrated, drifts between ZUPT resets)
float vx_mps = 0.0f;          // lateral/right velocity (integrated, drifts between ZUPT resets)
float yaw_rate_dps = 0.0f;    // direct gyro reading — NOT integrated, does not drift the same way
unsigned long last_sample_us = 0;
unsigned long still_since_ms = 0;
bool zupt_active = false;

uint8_t seq = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("Starting BNO08x IMU + ego-velocity firmware...");

  Serial2.begin(PI_UART_BAUD, SERIAL_8N1, PI_UART_RX_PIN, PI_UART_TX_PIN);

  Wire.begin();
  if (!bno08x.begin_I2C(BNO08X_I2C_ADDR, &Wire, BNO08X_INT)) {
    Serial.println("Failed to find BNO08x chip! Check wiring and I2C address.");
    while (1) { delay(10); }
  }
  Serial.println("BNO08x Found!");

  enableReports();
  Serial.printf("Calibrating bias (assume stationary) — %d samples...\n", CAL_SAMPLES);
}

void enableReports() {
  if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION, 20000)) {   // 20ms = 50Hz, gravity already removed
    Serial.println("Could not enable linear acceleration");
  }
  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 20000)) {
    Serial.println("Could not enable calibrated gyroscope");
  }
  // kept from the original sketch — not used by the velocity math below (see the mounting-
  // requirement note at the top), but useful for debugging orientation over Serial if needed later
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 20000)) {
    Serial.println("Could not enable rotation vector");
  }
}

uint8_t nmea_checksum(const String &body) {
  uint8_t cs = 0;
  for (size_t i = 0; i < body.length(); i++) cs ^= (uint8_t)body[i];
  return cs;
}

void send_egovel(float vx, float vy, float yaw_rate, uint8_t flags) {
  char body[64];
  // 2 decimal places matches the Pi-side parser's expectations (ego_velocity.py) and the
  // precision alerts.py already uses for its own sentences — kept consistent across the project
  snprintf(body, sizeof(body), "EGOVEL,%.2f,%.2f,%.2f,%u,%u", vx, vy, yaw_rate, seq, flags);
  String b(body);
  uint8_t cs = nmea_checksum(b);
  char line[80];
  snprintf(line, sizeof(line), "$%s*%02X\r\n", body, cs);
  Serial2.print(line);
  seq++;                                        // uint8_t: wraps 255 -> 0 on purpose, matches ego_velocity.py
}

void loop() {
  if (bno08x.wasReset()) {
    enableReports();
  }

  if (!bno08x.getSensorEvent(&sensorValue)) {
    return;
  }

  if (sensorValue.sensorId == SH2_GYROSCOPE_CALIBRATED) {
    // rad/s -> deg/s. Used both for the ZUPT stillness check below AND sent to the Pi directly —
    // this is the actual point of this file: the radar has no way to measure this on its own.
    yaw_rate_dps = sensorValue.un.gyroscope.z * 57.2958f;
    return;
  }

  if (sensorValue.sensorId == SH2_ROTATION_VECTOR) {
    // available for future use / debugging — not consumed by the velocity estimate (see the
    // mounting-requirement note at the top of this file)
    return;
  }

  if (sensorValue.sensorId != SH2_LINEAR_ACCELERATION) {
    return;
  }

  unsigned long now_us = micros();
  float ax = sensorValue.un.linearAcceleration.x;    // m/s^2, gravity already removed
  float ay = sensorValue.un.linearAcceleration.y;    // m/s^2, gravity already removed

  if (!calibrated) {
    bias_ax += ax;
    bias_ay += ay;
    cal_count++;
    if (cal_count >= CAL_SAMPLES) {
      bias_ax /= CAL_SAMPLES;
      bias_ay /= CAL_SAMPLES;
      calibrated = true;
      Serial.printf("Bias calibration done: bias_ax=%.4f bias_ay=%.4f m/s^2\n", bias_ax, bias_ay);
    }
    last_sample_us = now_us;
    return;                                          // don't integrate garbage before we know the bias
  }

  float dt = (last_sample_us == 0) ? 0.02f : (now_us - last_sample_us) * 1e-6f;
  last_sample_us = now_us;
  if (dt <= 0 || dt > 0.5f) dt = 0.02f;               // clock wrap / first sample / a long stall — skip, don't corrupt the integral

  float a_fwd = FORWARD_SIGN * (ax - bias_ax);
  float a_right = RIGHT_SIGN * (ay - bias_ay);

  bool still_now = (fabsf(a_fwd) < ACCEL_ZUPT_THRESH_MPS2) && (fabsf(a_right) < ACCEL_ZUPT_THRESH_MPS2) &&
                   (fabsf(yaw_rate_dps) < GYRO_ZUPT_THRESH_DPS);
  if (still_now) {
    if (still_since_ms == 0) still_since_ms = millis();
    if (millis() - still_since_ms >= ZUPT_HOLD_MS) {
      vy_mps = 0.0f;                                  // this is what actually bounds long-term drift
      vx_mps = 0.0f;
      zupt_active = true;
    }
  } else {
    still_since_ms = 0;
    zupt_active = false;
    vy_mps += a_fwd * dt;
    vx_mps += a_right * dt;
  }

  uint8_t flags = zupt_active ? 0x01 : 0x00;
  send_egovel(vx_mps, vy_mps, yaw_rate_dps, flags);

  // debug echo over USB — comment out for less serial-monitor noise once everything's working
  Serial.printf("vfwd=%.2f vlat=%.2f m/s  yaw=%.1f dps  a=(%.3f,%.3f)  %s\n",
                vy_mps, vx_mps, yaw_rate_dps, a_fwd, a_right, zupt_active ? "[ZUPT]" : "");
}
