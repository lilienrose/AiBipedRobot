#include "gait_sequencer.h"
#include "servo_control.h"
#include "config.h"

static GaitType currentGait = GAIT_NONE;
static int currentStepPhase = 0;
static unsigned long phaseStartTime = 0;
const unsigned long PHASE_HOLD_TIME = 150; // Pauza na stabilizaci po dosažení úhlů (ms)

// Definice 4 fáze jednoho cyklu chůze dopředu:
// [Levá kyčel, Levá koleno, Levý kotník, Pravá kyčel, Pravé koleno, Pravý kotník]
const int WALK_PHASES[4][NUM_SERVOS] = {
  {105,  90, 105,  90,  90,  90}, // Fáze 0: Náklon těžiště vlevo
  {105,  90, 105, 110, 110,  75}, // Fáze 1: Zvednutí a posun pravé nohy dopředu
  { 90,  90,  75,  75,  90,  75}, // Fáze 2: Náklon těžiště vpravo
  { 75, 110,  75,  75,  90,  75}  // Fáze 3: Zvednutí a posun levé nohy dopředu
};

void startGait(GaitType gait) {
  currentGait = gait;
  currentStepPhase = 0;
  phaseStartTime = millis();
  if (currentGait == GAIT_WALK_FORWARD) {
    setTargetAngles(WALK_PHASES[0]);
  }
}

void stopGait() {
  currentGait = GAIT_NONE;
  setHomePosition();
}

void updateGait() {
  if (currentGait == GAIT_NONE) return;

  // Vyčkáváme, až serva dodosáhnou cílových úhlů aktuální fáze
  if (areServosAtTarget()) {
    if (millis() - phaseStartTime >= PHASE_HOLD_TIME) {
      // Přechod na další fázi sekvence
      currentStepPhase = (currentStepPhase + 1) % 4;
      setTargetAngles(WALK_PHASES[currentStepPhase]);
      phaseStartTime = millis();
    }
  } else {
    phaseStartTime = millis(); // Reset časovače vyčkávání, dokud se serva pohybují
  }
}