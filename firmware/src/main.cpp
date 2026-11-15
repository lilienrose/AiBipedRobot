#include <Arduino.h>
#include "config.h"
#include "servo_control.h"
#include "robot_wifi.h"
#include "sensors.h"
#include "gait_sequencer.h"

unsigned long lastSensorRead = 0;
const int SENSOR_READ_INTERVAL = 200;

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n--- Spouštění robota ---");

  setupWiFi();
  setupServos();
  setupSensors();

  Serial.println("Robot je připraven.");
}

void loop() {
  handleUDP();
  updateGait();          // Aktualizace stavového stroje chůze
  updateServosSmooth();  // Plynulý posun serv po 3ms

  if (millis() - lastSensorRead >= SENSOR_READ_INTERVAL) {
    lastSensorRead = millis();
    
    float distance = readDistanceCM();
    if (distance > 0 && distance < 15.0) {
      Serial.printf("Překážka: %.1f cm -> Zastavuji chůzi\n", distance);
      stopGait(); // Bezpečnostní zastavení při detekci překážky
    }
  }
}