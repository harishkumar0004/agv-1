#include "Arduino.h"
#include "HardwareSerial.h"
#include "Motion.h"

Motion::Motion(Stepper& stepper, Imu& imu)
    : stepper(stepper),
      imu(imu) {
}

void Motion::begin() {
    command = NavigationCommand();

    headingErrorDeg = 0.0f;
    angularVelocity = 0.0f;

    baseDesiredHeadingDeg = 0.0f;

    initialLateralHeadingBiasDeg = 0.0f;
    activeLateralHeadingBiasDeg = 0.0f;

    activeTargetHeadingDeg = 0.0f;

    correctionDurationMs = 0.0f;
    correctionStartTimeMs = millis();

    lastCommandTime = millis();
    lastUpdateTime = micros();

    stepper.stop();
}

void Motion::update() {
    const uint32_t now = micros();

    if ((now - lastUpdateTime) < Config::CONTROL_PERIOD_US) {
        return;
    }

    lastUpdateTime = now;

    if ((millis() - lastCommandTime) > Config::COMMAND_TIMEOUT_MS) {
        stop();
        return;
    }

    float progress = 1.0f;

    if (correctionDurationMs > 0.0f) {
        const float elapsedMs =
            static_cast<float>(millis() - correctionStartTimeMs);

        progress = elapsedMs / correctionDurationMs;

        if (progress > 1.0f) {
            progress = 1.0f;
        }

        if (progress < 0.0f) {
            progress = 0.0f;
        }
    }

    activeLateralHeadingBiasDeg =
        initialLateralHeadingBiasDeg * (1.0f - progress);

    activeTargetHeadingDeg = normalizeAngle(
        baseDesiredHeadingDeg + activeLateralHeadingBiasDeg
    );

    headingErrorDeg = normalizeAngle(
        activeTargetHeadingDeg - imu.getHeading()
    );

    const float headingCorrection =
        Config::HEADING_KP * headingErrorDeg;

    angularVelocity =
        headingCorrection * DEG_TO_RAD;

    angularVelocity =
        Config::STEERING_DIRECTION *
        limitAngularVelocity(angularVelocity);

    stepper.setMotion(
        command.velocityMps,
        angularVelocity
    );
}

void Motion::setNavigationCommand(
    const NavigationCommand& navigationCommand) {

    command = navigationCommand;

    command.velocityMps =
        limitVelocity(command.velocityMps);

    command.desiredHeadingDeg =
        normalizeAngle(command.desiredHeadingDeg);

    baseDesiredHeadingDeg =
        command.desiredHeadingDeg;

    float correctionDistanceM =
        Config::TAG_SPACING_M *
        Config::LATERAL_CORRECTION_DISTANCE_RATIO;

    if (correctionDistanceM < 0.05f) {
        correctionDistanceM = 0.05f;
    }

    initialLateralHeadingBiasDeg =
        atan2f(command.lateralErrorM, correctionDistanceM) * RAD_TO_DEG;

    if (initialLateralHeadingBiasDeg > Config::MAX_LATERAL_HEADING_BIAS_DEG) {
        initialLateralHeadingBiasDeg = Config::MAX_LATERAL_HEADING_BIAS_DEG;
    }

    if (initialLateralHeadingBiasDeg < -Config::MAX_LATERAL_HEADING_BIAS_DEG) {
        initialLateralHeadingBiasDeg = -Config::MAX_LATERAL_HEADING_BIAS_DEG;
    }

    activeLateralHeadingBiasDeg =
        initialLateralHeadingBiasDeg;

    activeTargetHeadingDeg = normalizeAngle(
        baseDesiredHeadingDeg + activeLateralHeadingBiasDeg
    );

    const float velocityAbs =
        fabsf(command.velocityMps);

    if (velocityAbs < 0.01f) {
        correctionDurationMs = 0.0f;
    } else {
        correctionDurationMs =
            (correctionDistanceM / velocityAbs) * 1000.0f;
    }

    correctionStartTimeMs =
        millis();

    lastCommandTime =
        millis();
}

void Motion::stop() {
    command = NavigationCommand();

    headingErrorDeg = 0.0f;
    angularVelocity = 0.0f;

    baseDesiredHeadingDeg = 0.0f;

    initialLateralHeadingBiasDeg = 0.0f;
    activeLateralHeadingBiasDeg = 0.0f;

    activeTargetHeadingDeg = 0.0f;

    correctionDurationMs = 0.0f;
    correctionStartTimeMs = millis();

    stepper.stop();
}

NavigationCommand Motion::getNavigationCommand() const {
    return command;
}

float Motion::getHeadingError() const {
    return headingErrorDeg;
}

float Motion::getActiveTargetHeading() const {
    return activeTargetHeadingDeg;
}

float Motion::getActiveLateralBias() const {
    return activeLateralHeadingBiasDeg;
}

float Motion::getAngularVelocity() const {
    return angularVelocity;
}

float Motion::normalizeAngle(float angleDeg) {
    while (angleDeg > Config::HEADING_MAX_DEG) {
        angleDeg -= Config::FULL_CIRCLE_DEG;
    }

    while (angleDeg < Config::HEADING_MIN_DEG) {
        angleDeg += Config::FULL_CIRCLE_DEG;
    }

    return angleDeg;
}

float Motion::limitVelocity(float velocityMps) const {
    if (velocityMps > Config::MAX_LINEAR_VELOCITY_MPS) {
        return Config::MAX_LINEAR_VELOCITY_MPS;
    }

    if (velocityMps < -Config::MAX_LINEAR_VELOCITY_MPS) {
        return -Config::MAX_LINEAR_VELOCITY_MPS;
    }

    return velocityMps;
}

float Motion::limitAngularVelocity(float angularVelocity) const {
    if (angularVelocity > Config::MAX_STEERING) {
        return Config::MAX_STEERING;
    }

    if (angularVelocity < -Config::MAX_STEERING) {
        return -Config::MAX_STEERING;
    }

    return angularVelocity;
}