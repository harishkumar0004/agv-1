#pragma once

#include <Arduino.h>

#include "Imu.h"
#include "Stepper.h"

struct NavigationCommand {
    float velocityMps = 0.0f;
    float desiredHeadingDeg = 0.0f;
    float lateralErrorM = 0.0f;
};

class Motion {
public:
    Motion(Stepper& stepper, Imu& imu);

    void begin();
    void update();

    void stop();

    void setNavigationCommand(const NavigationCommand& command);

    NavigationCommand getNavigationCommand() const;

    float getHeadingError() const;
    float getActiveTargetHeading() const;
    float getActiveLateralBias() const;
    float getAngularVelocity() const;

private:
    static float normalizeAngle(float angleDeg);

    float limitVelocity(float velocityMps) const;
    float limitAngularVelocity(float angularVelocity) const;

private:
    Stepper& stepper;
    Imu& imu;

    NavigationCommand command;

    float headingErrorDeg = 0.0f;
    float angularVelocity = 0.0f;

    float baseDesiredHeadingDeg = 0.0f;

    float initialLateralHeadingBiasDeg = 0.0f;
    float activeLateralHeadingBiasDeg = 0.0f;

    float activeTargetHeadingDeg = 0.0f;

    float correctionDurationMs = 0.0f;
    uint32_t correctionStartTimeMs = 0;

    uint32_t lastCommandTime = 0;
    uint32_t lastUpdateTime = 0;
};