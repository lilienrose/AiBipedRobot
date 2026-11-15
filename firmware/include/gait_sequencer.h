#ifndef GAIT_SEQUENCER_H
#define GAIT_SEQUENCER_H

#include <Arduino.h>

enum GaitType {
  GAIT_NONE,
  GAIT_WALK_FORWARD
};

void startGait(GaitType gait);
void stopGait();
void updateGait();

#endif // GAIT_SEQUENCER_H