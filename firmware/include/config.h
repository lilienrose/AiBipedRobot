#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>
#include <IPAddress.h>

// Wi-Fi přístupové údaje
const char* const WIFI_SSID = "TVOJE_WIFI_NAZEV";
const char* const WIFI_PASSWORD = "TVOJE_WIFI_HESLO";

// Síťová konfigurace (UDP)
const uint16_t UDP_PORT = 5005;
const IPAddress ROBOT_IP(192, 168, 1, 150);
const IPAddress GATEWAY(192, 168, 1, 1);
const IPAddress SUBNET(255, 255, 255, 0);

// Hardware přiřazení GPIO pinů pro senzory
const int PIN_HC_TRIG = 5;
const int PIN_HC_ECHO = 17; // Přes dělič napětí (3.33 V)
const int PIN_MPU_SDA = 21;
const int PIN_MPU_SCL = 22;

// Hardware přiřazení GPIO pinů pro 6 servomotorů
// (Vybrány bezpečné PWM piny, vynechán GPIO 22 použitý pro I2C)
const int PIN_LEFT_HIP   = 13;
const int PIN_LEFT_KNEE  = 12;
const int PIN_LEFT_FOOT  = 14;

const int PIN_RIGHT_HIP  = 27;
const int PIN_RIGHT_KNEE = 26;
const int PIN_RIGHT_FOOT = 25;

// Počet serv pro pole
const int NUM_SERVOS = 6;
const int SERVO_PINS[NUM_SERVOS] = {
  PIN_LEFT_HIP, PIN_LEFT_KNEE, PIN_LEFT_FOOT,
  PIN_RIGHT_HIP, PIN_RIGHT_KNEE, PIN_RIGHT_FOOT
};

// Základní kalibrační úhly serv (Home / Stojící pozice)
const int HOME_LEFT_HIP   = 90;
const int HOME_LEFT_KNEE  = 90;
const int HOME_LEFT_FOOT  = 90;

const int HOME_RIGHT_HIP  = 90;
const int HOME_RIGHT_KNEE = 90;
const int HOME_RIGHT_FOOT = 90;

const int HOME_ANGLES[NUM_SERVOS] = {
  HOME_LEFT_HIP, HOME_LEFT_KNEE, HOME_LEFT_FOOT,
  HOME_RIGHT_HIP, HOME_RIGHT_KNEE, HOME_RIGHT_FOOT
};

const int SERVO_OFFSETS[NUM_SERVOS] = {0, 0, 0, 0, 0, 0};

#endif // CONFIG_H