#include "robot_wifi.h"
#include "config.h"
#include "servo_control.h"
#include "gait_sequencer.h"
#include <WiFi.h>
#include <WiFiUdp.h>

WiFiUDP udp;

// Statické pózy pro vybraná gesta
const int POSE_HOME[NUM_SERVOS]        = {90, 90, 90, 90, 90, 90};
const int POSE_LEAN_LEFT[NUM_SERVOS]   = {105, 90, 105, 90, 90, 90};   // OPEN_PALM
const int POSE_LEAN_RIGHT[NUM_SERVOS]  = {90, 90, 90, 75, 90, 75};     // PEACE
const int POSE_SQUAT[NUM_SERVOS]       = {90, 110, 70, 90, 110, 70};   // THUMBS_UP
const int POSE_TIPTOE[NUM_SERVOS]      = {90, 75, 105, 90, 75, 105};   // THUMBS_DOWN

void setupWiFi() {
  WiFi.mode(WIFI_STA);
  if (!WiFi.config(ROBOT_IP, GATEWAY, SUBNET)) {
    Serial.println("Chyba nastavení statické IP!");
  }
  
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Připojování k Wi-Fi");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWi-Fi připojeno!");
    Serial.print("IP adresa ESP32: ");
    Serial.println(WiFi.localIP());
    udp.begin(UDP_PORT);
    Serial.printf("UDP port %d otevřen.\n", UDP_PORT);
  } else {
    Serial.println("\nPřipojení k Wi-Fi selhalo. Běží offline.");
  }
}

void handleUDP() {
  int packetSize = udp.parsePacket();
  if (packetSize <= 0) return;

  char packetBuffer[255];
  int len = udp.read(packetBuffer, 255);
  if (len > 0) {
    packetBuffer[len] = 0;
  }

  String msg = String(packetBuffer);
  msg.trim();

  // Zpracování gest z Pythonu
  if (msg == "HOME" || msg == "FIST" || msg == "IDLE" || msg == "STOP") {
    stopGait();
    Serial.println("UDP -> Gesto: FIST / HOME (Zastavení)");
  }
  else if (msg == "POINTING" || msg == "FORWARD") {
    startGait(GAIT_WALK_FORWARD);
    Serial.println("UDP -> Gesto: POINTING (Spuštění chůze)");
  }
  else if (msg == "OPEN_PALM") {
    stopGait();
    setTargetAngles(POSE_LEAN_LEFT);
    Serial.println("UDP -> Gesto: OPEN_PALM (Náklon vlevo)");
  }
  else if (msg == "PEACE") {
    stopGait();
    setTargetAngles(POSE_LEAN_RIGHT);
    Serial.println("UDP -> Gesto: PEACE (Náklon vpravo)");
  }
  else if (msg == "THUMBS_UP") {
    stopGait();
    setTargetAngles(POSE_SQUAT);
    Serial.println("UDP -> Gesto: THUMBS_UP (Podřep)");
  }
  else if (msg == "THUMBS_DOWN") {
    stopGait();
    setTargetAngles(POSE_TIPTOE);
    Serial.println("UDP -> Gesto: THUMBS_DOWN (Vzpřímení)");
  }
  else {
    // Přímé nastavení úhlů z řetězce "90,90,90,90,90,90"
    int parsedAngles[NUM_SERVOS];
    int count = 0;
    int startIndex = 0;

    for (int i = 0; i <= msg.length(); i++) {
      if (i == msg.length() || msg.charAt(i) == ',') {
        if (count < NUM_SERVOS) {
          parsedAngles[count] = msg.substring(startIndex, i).toInt();
          count++;
        }
        startIndex = i + 1;
      }
    }

    if (count == NUM_SERVOS) {
      stopGait();
      setTargetAngles(parsedAngles);
      Serial.printf("Nové úhly: [%d, %d, %d, %d, %d, %d]\n",
                    parsedAngles[0], parsedAngles[1], parsedAngles[2],
                    parsedAngles[3], parsedAngles[4], parsedAngles[5]);
    }
  }
}