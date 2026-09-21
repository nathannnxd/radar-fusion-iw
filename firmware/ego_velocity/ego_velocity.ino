// ego_velocity.ino — ESP32 + Adafruit BNO08x IMU: measures the platform's own forward linear
// velocity and sends it to a Raspberry Pi over a dedicated hardware UART (GPIO pins, not the
// ESP32's USB port), for the radar-fusion pipeline's ego-motion compensation. See EGO_VELOCITY.md
// in the repo root for the full design, wiring diagram, and why this approach was chosen.
//
// Builds on the original gyro-test.ino (I2C setup, BNO08x init at 0x4B, INT/RESET pins) — same
// board pinout, extended with:
//   - SH2_LINEAR_ACCELERATION (gravity already removed by the BNO08x's own sensor fusion) and
//     SH2_GYROSCOPE_CALIBRATED, instead of just the rotation vector quaternion
//   - integrating linear acceleration into a velocity estimate, with two forms of drift
//     correction: a startup bias calibration, and a zero-velocity update (ZUPT) that pins the
//     estimate back to 0 whenever the platform is detected to be genuinely stationary
//   - sending the result out a SEPARATE hardware UART (Serial2) wired directly to the Pi's GPIO
//     UART pins, keeping the native Serial/USB port free for flashing and debug prints
//
// ---------------------------------------------------------------------------------------------
// IMPORTANT LIMITATION — read before trusting this for anything safety-critical
//
// This is dead-reckoning: velocity = integral of acceleration over time, with no independent
// ground-truth correction (no wheel encoder, no GPS, no vision odometry). Any residual
// accelerometer bias integrates into an ever-growing velocity error. The ZUPT logic below bounds
// that error TO ZERO every time the platform genuinely stops — which is why it matters — but
// between stops, on a long uninterrupted drive, expect real drift. This is good enough to correct
// the radar's Doppler ego-compensation (a few tens of a m/s of error there just nudges a
// static/moving point classification, not a hard safety threshold) — it is NOT a substitute for a
// wheel encoder or GPS if you need an accurate absolute speed over a long, continuous drive.
//
// ---------------------------------------------------------------------------------------------
// MOUNTING REQUIREMENT
//
// This firmware assumes the BNO08x's own +X axis is rigidly mounted pointing in the platform's
// direction of travel (forward). It does NOT use the rotation-vector quaternion to figure out
// "forward" dynamically — that's a deliberate simplification for this first version (see
// EGO_VELOCITY.md for the more robust version using the quaternion, left as a documented future
// improvement). Mount the board so the axis labeled X on the BNO08x/breakout points forward; if
// your breakout's silkscreen doesn't show which axis is X, log raw accelerometer values while
// pushing the board by hand in the forward direction and confirm which axis responds.

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

// -------------------------------------------------------------------------------- tuning
// Startup bias calibration: average this many samples (assumes the platform is stationary at
// power-on — if it isn't, the estimate will be off by whatever it was actually doing right then).
const int CAL_SAMPLES = 100;                  // ~2 s at the 50 Hz report rate below

// Zero-velocity update (ZUPT): the actual drift-bounding mechanism. If BOTH the (bias-corrected)
// forward acceleration and the yaw rate stay under these thresholds for ZUPT_HOLD_MS continuously,
// the platform is declared stationary and velocity is pinned to 0 rather than integrated —
// otherwise a small residual bias would make the "speed" climb forever while actually parked.
const float ACCEL_ZUPT_THRESH_MPS2 = 0.06f;   // m/s^2 — BNO08x linear-accel noise floor is roughly here
const float GYRO_ZUPT_THRESH_DPS   = 1.0f;    // deg/s — treat a slow turn as "not stationary" too
const unsigned long ZUPT_HOLD_MS   = 300;     // must stay under both thresholds this long before zeroing

// -------------------------------------------------------------------------------- state
float bias_ax = 0.0f;
int cal_count = 0;
bool calibrated = false;

float velocity_mps = 0.0f;
float last_gyro_z_dps = 0.0f;
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

void send_egovel(float speed_mps, uint8_t flags) {
  char body[48];
  // 2 decimal places matches the Pi-side parser's expectations (ego_velocity.py) and the
  // precision alerts.py already uses for its own sentences — kept consistent across the project
  snprintf(body, sizeof(body), "EGOVEL,%.2f,%u,%u", speed_mps, seq, flags);
  String b(body);
  uint8_t cs = nmea_checksum(b);
  char line[64];
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
    // rad/s -> deg/s, z axis = yaw rate (used only for the ZUPT stillness check, not integrated
    // into heading — no attempt at full dead-reckoning position/orientation here, speed only)
    last_gyro_z_dps = sensorValue.un.gyroscope.z * 57.2958f;
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
  float ax = sensorValue.un.linearAcceleration.x;    // m/s^2, gravity already removed, +X = forward (by mounting)

  if (!calibrated) {
    bias_ax += ax;
    cal_count++;
    if (cal_count >= CAL_SAMPLES) {
      bias_ax /= CAL_SAMPLES;
      calibrated = true;
      Serial.printf("Bias calibration done: bias_ax = %.4f m/s^2\n", bias_ax);
    }
    last_sample_us = now_us;
    return;                                          // don't integrate garbage before we know the bias
  }

  float dt = (last_sample_us == 0) ? 0.02f : (now_us - last_sample_us) * 1e-6f;
  last_sample_us = now_us;
  if (dt <= 0 || dt > 0.5f) dt = 0.02f;               // clock wrap / first sample / a long stall — skip, don't corrupt the integral

  float a_corrected = ax - bias_ax;

  bool still_now = (fabsf(a_corrected) < ACCEL_ZUPT_THRESH_MPS2) && (fabsf(last_gyro_z_dps) < GYRO_ZUPT_THRESH_DPS);
  if (still_now) {
    if (still_since_ms == 0) still_since_ms = millis();
    if (millis() - still_since_ms >= ZUPT_HOLD_MS) {
      velocity_mps = 0.0f;                            // this is what actually bounds long-term drift
      zupt_active = true;
    }
  } else {
    still_since_ms = 0;
    zupt_active = false;
    velocity_mps += a_corrected * dt;
  }

  uint8_t flags = zupt_active ? 0x01 : 0x00;
  send_egovel(velocity_mps, flags);

  // debug echo over USB — comment out for less serial-monitor noise once everything's working
  Serial.printf("v=%.2f m/s  a=%.3f  gz=%.1f dps  %s\n", velocity_mps, a_corrected, last_gyro_z_dps,
                zupt_active ? "[ZUPT]" : "");
}
