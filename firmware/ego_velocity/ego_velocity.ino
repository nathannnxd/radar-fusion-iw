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
// Ego-motion extension (EGO_MOTION.md §2, §6b) — the sentence is now the 10-field "v2":
//   $EGOVEL,<vx>,<vy>,<yaw_rate_dps>,<seq>,<flags>,<esp_ms>,<pitch_deg>,<roll_deg>,<acc_fwd_mps2>,<gyro_cal>*<XOR>
//   - SH2_GAME_ROTATION_VECTOR (50 Hz, no magnetometer: the steel chassis makes the magnetic one
//     useless) gives pitch/roll for the Pi's slope handling; derived from the quaternion below
//   - esp_ms = millis() at the linear-acceleration sample; the Pi maps it onto its own clock with a
//     running median offset, so USB/UART jitter no longer smears the sample time
//   - acc_fwd = the bias-corrected forward acceleration the ZUPT already uses
//   - gyro_cal = the BNO08x gyroscope accuracy status (0..3, sensorValue.status) so the Pi can
//     refuse an uncalibrated gyro (gyro_ok needs >= 2)
//   - ALERT_NODE 1 folds esp32/alert_node/alert_node.ino into this sketch: the same board listens
//     for the Pi's $RDALT sentences on the USB Serial and drives the LEDs + buzzer, while the IMU
//     stream goes out Serial2 — one ESP32, both jobs (EGO_MOTION.md §1). Set it to 0 to build the
//     plain IMU sketch (Serial stays a debug console only).
//   The Pi-side reader (ego_velocity.py) accepts both the old 5-field v1 and this v2 sentence.
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
// rather than trying to physically remount the board. Yaw needs the same check and does NOT follow
// from those two: X-forward/Y-right is a right-handed frame whose +Z points DOWN, so the raw gyro z
// reads POSITIVE ON A RIGHT TURN while the $EGOVEL contract wants left-turn positive (EGO_MOTION.md
// §2) — YAW_SIGN below is what converts it, and it is its own knob.

#include <Wire.h>
#include <Adafruit_BNO08x.h>

// -------------------------------------------------------------------------------- build options
// 1: also run the driver-alert node (esp32/alert_node/alert_node.ino) on this board — $RDALT in over the
// USB Serial, LEDs + buzzer out (pins below). 0: IMU link only. See EGO_MOTION.md §1 / §6b.
#define ALERT_NODE 1

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
// See the MOUNTING REQUIREMENT note above — adjust these three after physically confirming your
// board's actual orientation. A wrong sign here silently flips "closing" vs "receding" (FORWARD_SIGN),
// "left" vs "right" (RIGHT_SIGN) or "turning left" vs "turning right" (YAW_SIGN) in every downstream
// ego-compensated value.
#define FORWARD_SIGN 1.0f   // BNO08x +X assumed forward
#define RIGHT_SIGN   1.0f   // BNO08x +Y assumed right
// With the documented mounting (+X forward, +Y right) the chip's +Z points DOWN, so raw gyro z is
// right-turn positive; $EGOVEL wants + = CCW from above = turning left (EGO_MOTION.md §2), hence -1.
// Use +1.0f only on a Z-up mount (+X forward, +Y LEFT). Determine it before any lever-arm calibration:
// yaw the tractor left by hand and confirm yaw_rate_dps reads positive (tools/drills.md, pre-drive).
#define YAW_SIGN     -1.0f  // raw gyro z (right-turn +) -> contract yaw rate (left-turn +)

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
uint8_t gyro_cal = 0;         // BNO08x gyroscope accuracy status 0..3 (sensorValue.status bits 1-0), sent as gyro_cal
float pitch_deg = 0.0f;       // from the Game Rotation Vector: nose-up +  (EGO_MOTION.md §2)
float roll_deg = 0.0f;        // idem: right side down +
bool attitude_known = false;  // no game-rotation report yet -> pitch/roll fields are sent empty ("unknown")
unsigned long last_sample_us = 0;
unsigned long still_since_ms = 0;
bool zupt_active = false;

uint8_t seq = 0;

// -------------------------------------------------------------------------------- alert node (optional)
#if ALERT_NODE
// Verbatim logic of esp32/alert_node/alert_node.ino: three LEDs + a buzzer driven by the Pi's $RDALT
// sentences (ALERTS.md) arriving on the USB Serial. Fail-visible: if the Pi stops talking, everything
// slow-blinks instead of staying green. The IMU stream on Serial2 is unaffected.
// Wiring: LEDs through 220 Ω to GND, active buzzer (+) to BUZZER, (−) to GND. None of these pins collide
// with the IMU (I2C 21/22, INT 4, RESET 15) or the Pi UART (16/17).
const int LED_LINK   = 2;     // onboard LED on most DevKits: blinks on every received sentence
const int LED_GREEN  = 26;    // link OK, no alert
const int LED_YELLOW = 27;    // warning (110, 101, 103, 120, 121, 130, 131, 141, or any 'W')
const int LED_RED    = 14;    // critical (111, 122 or any 'C')
const int BUZZER     = 25;    // active buzzer (just on/off). For a passive one use tone()/noTone() instead.
// const int RELAY_STOP = 33; // optional: a relay / opto to the tractor's stop input on critical — see alert_node_loop()

const unsigned long LINK_TIMEOUT_MS  = 5000;   // no sentence for this long → Pi/cable dead → slow-blink everything
const unsigned long ALERT_TIMEOUT_MS = 6000;   // an active alert not resent for this long is considered cleared
const int           MAX_ALERTS       = 16;

struct Alert { int code; long obj; char sev; unsigned long last; bool used; };
Alert alerts[MAX_ALERTS];

unsigned long lastLineMs = 0;
char alert_line[160]; int alert_line_len = 0;

bool checksumOk(const char *s) {                 // s = "$RDALT,...*7B"
  if (s[0] != '$') return false;
  const char *star = strchr(s, '*');
  if (!star || strlen(star) < 3) return false;
  uint8_t cs = 0;
  for (const char *p = s + 1; p < star; p++) cs ^= (uint8_t)*p;
  return cs == (uint8_t)strtol(star + 1, NULL, 16);
}

void rememberAlert(int code, char event, char sev, long obj) {
  int free_ = -1;
  for (int i = 0; i < MAX_ALERTS; i++) {
    if (alerts[i].used && alerts[i].code == code && alerts[i].obj == obj) {
      if (event == 'C') alerts[i].used = false;
      else { alerts[i].sev = sev; alerts[i].last = millis(); }
      return;
    }
    if (!alerts[i].used && free_ < 0) free_ = i;
  }
  if (event != 'C' && free_ >= 0) alerts[free_] = { code, obj, sev, millis(), true };
}

void onAlertLine(char *s) {
  if (strncmp(s, "$RDALT,", 7) != 0 || !checksumOk(s)) return;
  *strchr(s, '*') = 0;                            // drop the checksum, then split on ','
  char *f[9]; int n = 0;
  for (char *tok = strtok(s + 1, ","); tok && n < 9; tok = strtok(NULL, ",")) f[n++] = tok;
  if (n < 5) return;
  int code = atoi(f[1]); char event = f[2][0]; char sev = f[3][0]; long obj = atol(f[4]);
  rememberAlert(code, event, sev, obj);
  lastLineMs = millis();
  digitalWrite(LED_LINK, !digitalRead(LED_LINK));
  Serial.printf("ok code=%d event=%c sev=%c obj=%ld\n", code, event, sev, obj);   // echo for debugging (harmless for the Pi)
}

char currentLevel() {                            // 'C' > 'W' > 'I' > '-' among alerts still alive
  char lvl = '-'; unsigned long now = millis();
  for (int i = 0; i < MAX_ALERTS; i++) {
    if (!alerts[i].used) continue;
    if (now - alerts[i].last > ALERT_TIMEOUT_MS) { alerts[i].used = false; continue; }
    char s = alerts[i].sev;
    if (alerts[i].code == 140) s = 'W';          // radar stale: the system is half blind — treat as a warning
    if (s == 'C' || (s == 'W' && lvl != 'C') || (s == 'I' && lvl == '-')) lvl = s;
  }
  return lvl;
}

void alert_node_setup() {
  const int pins[] = { LED_LINK, LED_GREEN, LED_YELLOW, LED_RED, BUZZER };
  for (int p : pins) { pinMode(p, OUTPUT); digitalWrite(p, LOW); }
  // pinMode(RELAY_STOP, OUTPUT); digitalWrite(RELAY_STOP, LOW);
  Serial.println("alert_node ready");
}

void alert_node_loop() {                         // called every loop() pass, independent of IMU events
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') { if (alert_line_len) { alert_line[alert_line_len] = 0; onAlertLine(alert_line); } alert_line_len = 0; }
    else if (alert_line_len < (int)sizeof(alert_line) - 1) alert_line[alert_line_len++] = c;
  }

  unsigned long now = millis();
  bool linkLost = (now - lastLineMs > LINK_TIMEOUT_MS);
  char lvl = linkLost ? '-' : currentLevel();
  bool blinkSlow = (now / 500) % 2, blinkFast = (now / 120) % 2;

  if (linkLost) {                                 // Pi silent: everything blinks slowly, buzzer chirps once a second
    digitalWrite(LED_GREEN, blinkSlow); digitalWrite(LED_YELLOW, blinkSlow); digitalWrite(LED_RED, blinkSlow);
    digitalWrite(BUZZER, (now % 1000) < 60);
  } else if (lvl == 'C') {                        // critical: red + fast beep
    digitalWrite(LED_GREEN, LOW); digitalWrite(LED_YELLOW, LOW); digitalWrite(LED_RED, HIGH);
    digitalWrite(BUZZER, blinkFast);
    // digitalWrite(RELAY_STOP, HIGH);
  } else if (lvl == 'W') {                        // warning: yellow + slow beep
    digitalWrite(LED_GREEN, LOW); digitalWrite(LED_YELLOW, HIGH); digitalWrite(LED_RED, LOW);
    digitalWrite(BUZZER, blinkSlow);
    // digitalWrite(RELAY_STOP, LOW);
  } else {                                        // info or nothing: green, quiet
    digitalWrite(LED_GREEN, HIGH); digitalWrite(LED_YELLOW, lvl == 'I'); digitalWrite(LED_RED, LOW);
    digitalWrite(BUZZER, LOW);
    // digitalWrite(RELAY_STOP, LOW);
  }
}
#endif  // ALERT_NODE

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("Starting BNO08x IMU + ego-velocity firmware...");

  Serial2.begin(PI_UART_BAUD, SERIAL_8N1, PI_UART_RX_PIN, PI_UART_TX_PIN);
#if ALERT_NODE
  alert_node_setup();
#endif

  Wire.begin();
  if (!bno08x.begin_I2C(BNO08X_I2C_ADDR, &Wire, BNO08X_INT)) {
    Serial.println("Failed to find BNO08x chip! Check wiring and I2C address.");
    while (1) {
#if ALERT_NODE
      alert_node_loop();                          // a dead IMU must not also kill the driver's alert lights
#endif
      delay(10);
    }
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
  // Game Rotation Vector: accel + gyro fusion only, no magnetometer (the tractor is a steel chassis —
  // the magnetic rotation vector would wander). Only pitch/roll are derived from it; heading from it
  // drifts slowly and is not sent. Replaces the debug-only SH2_ROTATION_VECTOR of the first version.
  if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 20000)) {   // 50 Hz, EGO_MOTION.md §6b
    Serial.println("Could not enable game rotation vector");
  }
}

uint8_t nmea_checksum(const String &body) {
  uint8_t cs = 0;
  for (size_t i = 0; i < body.length(); i++) cs ^= (uint8_t)body[i];
  return cs;
}

void send_egovel(float vx, float vy, float yaw_rate, uint8_t flags, unsigned long esp_ms, float a_fwd) {
  char body[112];
  // v2 sentence (EGO_MOTION.md §2): the first five fields are exactly the v1 sentence, so an older Pi-side
  // reader that only checks them keeps working. 2 decimal places on velocities matches ego_velocity.py and
  // the precision alerts.py already uses; pitch/roll get one, the accel three (the ZUPT threshold is 0.06).
  // Empty pitch/roll fields = "unknown" until the first game-rotation report has arrived.
  char att[24] = ",";                          // ",<pitch>,<roll>" or "," + "" (two empty fields)
  if (attitude_known) snprintf(att, sizeof(att), "%.1f,%.1f", pitch_deg, roll_deg);
  snprintf(body, sizeof(body), "EGOVEL,%.2f,%.2f,%.2f,%u,%u,%lu,%s,%.3f,%u",
           vx, vy, yaw_rate, seq, flags, esp_ms, att, a_fwd, gyro_cal);
  String b(body);
  uint8_t cs = nmea_checksum(b);
  char line[128];
  snprintf(line, sizeof(line), "$%s*%02X\r\n", body, cs);
  Serial2.print(line);
  seq++;                                        // uint8_t: wraps 255 -> 0 on purpose, matches ego_velocity.py
}

void update_attitude(float qi, float qj, float qk, float qw) {
  // World "up" expressed in the sensor frame = third row of the rotation matrix of the (unit) quaternion.
  // pitch = tilt of the forward axis above the horizon (nose-up +), roll = tilt of the right axis below it
  // (right side down +): asin of the axis' component along "up" — independent of whether the board's Z points
  // up or down when mounted, and FORWARD_SIGN / RIGHT_SIGN flip them together with the accelerations.
  float ux = 2.0f * (qi * qk - qw * qj);
  float uy = 2.0f * (qj * qk + qw * qi);
  ux = constrain(ux, -1.0f, 1.0f);
  uy = constrain(uy, -1.0f, 1.0f);
  pitch_deg = FORWARD_SIGN * asinf(ux) * 57.2958f;
  roll_deg  = RIGHT_SIGN * -asinf(uy) * 57.2958f;
  attitude_known = true;
}

void loop() {
#if ALERT_NODE
  alert_node_loop();                            // LEDs/buzzer + $RDALT parsing, every pass, even with no IMU event
#endif

  if (bno08x.wasReset()) {
    enableReports();
  }

  if (!bno08x.getSensorEvent(&sensorValue)) {
    return;
  }

  if (sensorValue.sensorId == SH2_GYROSCOPE_CALIBRATED) {
    // rad/s -> deg/s. Used both for the ZUPT stillness check below AND sent to the Pi directly —
    // this is the actual point of this file: the radar has no way to measure this on its own.
    yaw_rate_dps = YAW_SIGN * sensorValue.un.gyroscope.z * 57.2958f;
    gyro_cal = sensorValue.status & 0x03;      // SH-2 accuracy: 0 unreliable, 1 low, 2 medium, 3 high
    return;
  }

  if (sensorValue.sensorId == SH2_GAME_ROTATION_VECTOR) {
    update_attitude(sensorValue.un.gameRotationVector.i, sensorValue.un.gameRotationVector.j,
                    sensorValue.un.gameRotationVector.k, sensorValue.un.gameRotationVector.real);
    return;
  }

  if (sensorValue.sensorId != SH2_LINEAR_ACCELERATION) {
    return;
  }

  unsigned long now_us = micros();
  unsigned long sample_ms = millis();              // esp_ms of this sentence: the Pi maps it onto its own clock
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
  send_egovel(vx_mps, vy_mps, yaw_rate_dps, flags, sample_ms, a_fwd);

#if !ALERT_NODE
  // debug echo over USB — comment out for less serial-monitor noise once everything's working.
  // Off when the alert node shares this Serial: 50 lines/s of echo next to the $RDALT traffic is
  // just noise on the Pi side (harmless — alerts.py never reads the port — but pointless).
  Serial.printf("vfwd=%.2f vlat=%.2f m/s  yaw=%.1f dps (cal %u)  pitch=%.1f roll=%.1f  a=(%.3f,%.3f)  %s\n",
                vy_mps, vx_mps, yaw_rate_dps, gyro_cal, pitch_deg, roll_deg, a_fwd, a_right, zupt_active ? "[ZUPT]" : "");
#endif
}
