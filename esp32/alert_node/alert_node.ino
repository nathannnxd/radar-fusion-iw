// alert_node.ino — ESP32 driver-alert node for the radar+camera fusion (Team 3).
//
// Listens on USB-serial for the Pi's $RDALT sentences (see ALERTS.md in the repo) and turns them into
// something a driver notices without looking at a screen: three LEDs + a buzzer. Fail-visible: if the
// Pi stops talking, everything slow-blinks instead of staying green.
//
//   $RDALT,<code>,<event>,<severity>,<obj_id>,<range_m>,<az_deg>,<speed_mps>,<flags>*<checksum>\r\n
//   event: S start, A active (resent every ~2 s), C clear · severity: I info, W warning, C critical
//   codes: 100 radar-only object, 101 camera lost (radar hold), 102 out of frame, 103 camera degraded,
//          110 proximity warning, 111 proximity critical, 120 fast approach, 140 radar sensor stale
//
// Wiring (any GPIO works, change below): LEDs through 220 Ω to GND, active buzzer (+) to BUZZER pin, (−) to GND.
// Pi side: configs.json → "ALERT_SERIAL_PORT": "/dev/ttyUSB0" (CP2102/CH340 boards) or "/dev/ttyACM2" (S3/C3 native USB),
//          "ALERT_BAUD": 115200. Plug the ESP32 in AFTER the radar, or use /dev/serial/by-id/... names, so the
//          radar's ttyACM0/ttyACM1 numbering doesn't shift.

#include <Arduino.h>

const int LED_LINK   = 2;     // onboard LED on most DevKits: blinks on every received sentence
const int LED_GREEN  = 26;    // link OK, no alert
const int LED_YELLOW = 27;    // warning (110, 101, 103, 120, or any 'W')
const int LED_RED    = 14;    // critical (111 or any 'C')
const int BUZZER     = 25;    // active buzzer (just on/off). For a passive one use tone()/noTone() instead.
// const int RELAY_STOP = 33; // optional: a relay / opto to the tractor's stop input on critical — see loop()

const unsigned long LINK_TIMEOUT_MS  = 5000;   // no sentence for this long → Pi/cable dead → slow-blink everything
const unsigned long ALERT_TIMEOUT_MS = 6000;   // an active alert not resent for this long is considered cleared
const int           MAX_ALERTS       = 16;

struct Alert { int code; long obj; char sev; unsigned long last; bool used; };
Alert alerts[MAX_ALERTS];

unsigned long lastLineMs = 0;
char line[160]; int lineLen = 0;

// ---------------------------------------------------------------- parsing
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

void onLine(char *s) {
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

// ---------------------------------------------------------------- state → outputs
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

void setup() {
  Serial.begin(115200);
  const int pins[] = { LED_LINK, LED_GREEN, LED_YELLOW, LED_RED, BUZZER };
  for (int p : pins) { pinMode(p, OUTPUT); digitalWrite(p, LOW); }
  // pinMode(RELAY_STOP, OUTPUT); digitalWrite(RELAY_STOP, LOW);
  Serial.println("alert_node ready");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') { if (lineLen) { line[lineLen] = 0; onLine(line); } lineLen = 0; }
    else if (lineLen < (int)sizeof(line) - 1) line[lineLen++] = c;
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
