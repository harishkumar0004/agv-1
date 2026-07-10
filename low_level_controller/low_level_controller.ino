#include <Arduino.h>
#include <Wire.h>
#include <string.h>
#include <math.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

constexpr uint8_t LEFT_STEP_PIN = 16;
constexpr uint8_t LEFT_DIR_PIN  = 26;
constexpr uint8_t LEFT_EN_PIN   = 25;

constexpr uint8_t RIGHT_STEP_PIN = 4;
constexpr uint8_t RIGHT_DIR_PIN  = 13;
constexpr uint8_t RIGHT_EN_PIN   = 14;

constexpr uint8_t DRIVER_ENABLE_LEVEL  = LOW;
constexpr uint8_t DRIVER_DISABLE_LEVEL = HIGH;

// Timers
constexpr uint32_t INITIAL_TIMER_PERIOD_US = 1000;

// Robot geometry
constexpr float PI_F = 3.14159265358979323846f;
constexpr float DEG_TO_RAD_F = PI_F / 180.0f;
constexpr float RAD_TO_DEG_F = 180.0f / PI_F;

constexpr float WHEEL_DIAMETER_M = 0.117f;
constexpr uint32_t STEPS_PER_WHEEL_REV = 20000;
constexpr float WHEEL_CIRCUMFERENCE_M = PI_F * WHEEL_DIAMETER_M;
constexpr float STEPS_PER_METER = STEPS_PER_WHEEL_REV / WHEEL_CIRCUMFERENCE_M;

constexpr float TRACK_WIDTH_M = 0.355f;

// Motion limits
constexpr float MAX_LINEAR_VELOCITY_MPS = 0.10f;
constexpr float MAX_STEP_RATE = 5000.0f;
constexpr float MIN_STEP_RATE = 1.0f;

// Simple trapezoidal/ramp acceleration
constexpr float LINEAR_ACCEL_MPS2 = 0.08f;

// MPU6050
constexpr uint8_t MPU6050_ADDRESS = 0x68;
constexpr uint8_t REG_PWR_MGMT_1  = 0x6B;
constexpr uint8_t REG_GYRO_CONFIG = 0x1B;
constexpr uint8_t REG_GYRO_ZOUT_H = 0x47;

constexpr float GYRO_SCALE_250DPS = 131.0f;
constexpr uint16_t IMU_CALIBRATION_SAMPLES = 1000;
constexpr uint16_t IMU_CALIBRATION_DELAY_MS = 2;

constexpr float GYRO_DEADBAND_DPS = 0.25f;

// Heading
constexpr float FULL_CIRCLE_DEG = 360.0f;
constexpr float HEADING_MAX_DEG = 180.0f;
constexpr float HEADING_MIN_DEG = -180.0f;

// Controller
constexpr float HEADING_KP = 0.40f;

constexpr float TAG_SPACING_M = 0.50f;
constexpr float LATERAL_CORRECTION_DISTANCE_RATIO = 0.70f;
constexpr float LATERAL_GAIN = 1.0f;

constexpr float MAX_LATERAL_HEADING_BIAS_DEG = 5.0f;
constexpr float MAX_STEERING = 0.06f;
constexpr float STEERING_DIRECTION = 1.0f;

// Timing
constexpr uint32_t CONTROL_PERIOD_US = 2500;
constexpr uint32_t COMMAND_TIMEOUT_MS = 30000;
constexpr uint32_t STATUS_PERIOD_MS = 250;

// Turning
constexpr float TURN_KP = 0.5f;
constexpr float TURN_MAX_WZ = 0.45f;
constexpr float TURN_MIN_WZ = 0.12f;
constexpr float TURN_TOLERANCE_DEG = 1.0f;
constexpr uint32_t TURN_TIMEOUT_MS = 10000;

constexpr float TURN_DIRECTION = 1.0f;

// for counting how many distace the agv travelled
volatile int64_t leftStepPulseCount = 0;
volatile int64_t rightStepPulseCount = 0;

int64_t correctionStartLeftPulseCount = 0;
int64_t correctionStartRightPulseCount = 0;

float correctionDistanceTargetM = 0.0f;
float correctionDistanceTravelledM = 0.0f;
// ============================================================================
// GLOBAL STATE
// ============================================================================

// Stepper
hw_timer_t* leftTimer = nullptr;
hw_timer_t* rightTimer = nullptr;

volatile bool leftStepState = false;
volatile bool rightStepState = false;

bool motorsEnabled = false;

float leftVelocityMps = 0.0f;
float rightVelocityMps = 0.0f;
float leftStepRatePps = 0.0f;
float rightStepRatePps = 0.0f;

// Turn state
bool turnActive = false;
float turnTargetHeadingDeg = 0.0f;
uint32_t turnStartTimeMs = 0;

// IMU
float imuHeadingDeg = 0.0f;
float gyroRateDegPerSec = 0.0f;
float gyroBiasDegPerSec = 0.0f;
int16_t rawGyroZ = 0;
uint32_t lastImuUpdateMicros = 0;
bool imuCalibrated = false;

// Motion mode
enum MotionMode {
    MODE_STOP = 0,
    MODE_NORMAL = 1,
    MODE_APPROACH = 2,
};

MotionMode motionMode = MODE_STOP;

// Motion command
float commandVelocityMps = 0.0f;
float commandDesiredHeadingDeg = 0.0f;

float commandXLateralErrorM = 0.0f;
float commandYLateralErrorM = 0.0f;

float rampedVelocityMps = 0.0f;

float baseDesiredHeadingDeg = 0.0f;
float initialLateralHeadingBiasDeg = 0.0f;
float activeLateralHeadingBiasDeg = 0.0f;
float activeTargetHeadingDeg = 0.0f;

float headingErrorDeg = 0.0f;
float angularVelocityRadS = 0.0f;

float correctionDurationMs = 0.0f;
uint32_t correctionStartTimeMs = 0;

uint32_t lastMotionUpdateMicros = 0;
uint32_t lastCommandTimeMs = 0;
uint32_t lastStatusTimeMs = 0;

bool commandActive = false;
bool calibrating = false;

// Serial command buffer
char commandBuffer[128];
uint8_t commandIndex = 0;

// Helper function for getting distance or pulse to genertae the left and right motor

float getCorrectionTravelledDistanceM() {
    noInterrupts();

    const int64_t leftNow = leftStepPulseCount;
    const int64_t rightNow = rightStepPulseCount;

    interrupts();

    const int64_t leftDelta =
        llabs(leftNow - correctionStartLeftPulseCount);

    const int64_t rightDelta =
        llabs(rightNow - correctionStartRightPulseCount);

    const float averagePulses =
        0.5f * static_cast<float>(leftDelta + rightDelta);

    return averagePulses / STEPS_PER_METER;
}

// ============================================================================
// UTILITY
// ============================================================================

float normalizeAngle(float angleDeg) {
    while (angleDeg > HEADING_MAX_DEG) {
        angleDeg -= FULL_CIRCLE_DEG;
    }

    while (angleDeg <= HEADING_MIN_DEG) {
        angleDeg += FULL_CIRCLE_DEG;
    }

    return angleDeg;
}

float clampFloat(float value, float low, float high) {
    if (value < low) {
        return low;
    }

    if (value > high) {
        return high;
    }

    return value;
}

float limitVelocity(float velocityMps) {
    return clampFloat(
        velocityMps,
        -MAX_LINEAR_VELOCITY_MPS,
        MAX_LINEAR_VELOCITY_MPS
    );
}

float limitAngularVelocity(float angularVelocity) {
    return clampFloat(
        angularVelocity,
        -MAX_STEERING,
        MAX_STEERING
    );
}

float rampToward(float current, float target, float maxChange) {
    if (current < target) {
        current += maxChange;

        if (current > target) {
            current = target;
        }
    } else if (current > target) {
        current -= maxChange;

        if (current < target) {
            current = target;
        }
    }

    return current;
}


// ============================================================================
// STEPPER FUNCTIONS
// ============================================================================

void IRAM_ATTR onLeftTimer() {
    leftStepState = !leftStepState;
    digitalWrite(LEFT_STEP_PIN, leftStepState);

    if(leftStepState){
        leftStepPulseCount++;
    }
}

void IRAM_ATTR onRightTimer() {
    rightStepState = !rightStepState;
    digitalWrite(RIGHT_STEP_PIN, rightStepState);

    if (rightStepState) {
        rightStepPulseCount++;
    }
}

float velocityToStepRate(float velocityMps) {
    return velocityMps * STEPS_PER_METER;
}

void applyLeftStepRate(float stepRate) {
    leftStepRatePps = stepRate;

    if (fabsf(stepRate) < MIN_STEP_RATE) {
        leftStepRatePps = 0.0f;
        timerStop(leftTimer);
        digitalWrite(LEFT_STEP_PIN, LOW);
        leftStepState = false;
        return;
    }

    digitalWrite(LEFT_DIR_PIN, stepRate >= 0.0f);

    stepRate = fabsf(stepRate);

    if (stepRate > MAX_STEP_RATE) {
        stepRate = MAX_STEP_RATE;
    }

    const uint32_t periodUs =
        static_cast<uint32_t>(1000000.0f / (2.0f * stepRate));

    timerAlarm(leftTimer, periodUs, true, 0);
    timerStart(leftTimer);
}

void applyRightStepRate(float stepRate) {
    rightStepRatePps = stepRate;

    if (fabsf(stepRate) < MIN_STEP_RATE) {
        rightStepRatePps = 0.0f;
        timerStop(rightTimer);
        digitalWrite(RIGHT_STEP_PIN, LOW);
        rightStepState = false;
        return;
    }

    digitalWrite(RIGHT_DIR_PIN, stepRate >= 0.0f);

    stepRate = fabsf(stepRate);

    if (stepRate > MAX_STEP_RATE) {
        stepRate = MAX_STEP_RATE;
    }

    const uint32_t periodUs =
        static_cast<uint32_t>(1000000.0f / (2.0f * stepRate));

    timerAlarm(rightTimer, periodUs, true, 0);
    timerStart(rightTimer);
}

void setLeftVelocity(float velocityMps) {
    leftVelocityMps = velocityMps;
    applyLeftStepRate(velocityToStepRate(velocityMps));
}

void setRightVelocity(float velocityMps) {
    rightVelocityMps = velocityMps;
    applyRightStepRate(velocityToStepRate(velocityMps));
}

void stopMotors() {
    leftVelocityMps = 0.0f;
    rightVelocityMps = 0.0f;
    leftStepRatePps = 0.0f;
    rightStepRatePps = 0.0f;

    timerStop(leftTimer);
    timerStop(rightTimer);

    digitalWrite(LEFT_STEP_PIN, LOW);
    digitalWrite(RIGHT_STEP_PIN, LOW);

    leftStepState = false;
    rightStepState = false;
}

void enableMotors() {
    digitalWrite(LEFT_EN_PIN, DRIVER_ENABLE_LEVEL);
    digitalWrite(RIGHT_EN_PIN, DRIVER_ENABLE_LEVEL);
    motorsEnabled = true;
}

void disableMotors() {
    stopMotors();

    digitalWrite(LEFT_EN_PIN, DRIVER_DISABLE_LEVEL);
    digitalWrite(RIGHT_EN_PIN, DRIVER_DISABLE_LEVEL);

    motorsEnabled = false;
}

void setMotion(float linearVelocityMps, float angularVelocity) {
    if (!motorsEnabled) {
        return;
    }

    const float halfTrack = TRACK_WIDTH_M * 0.5f;

    const float leftVelocity =
        linearVelocityMps - (angularVelocity * halfTrack);

    const float rightVelocity =
        linearVelocityMps + (angularVelocity * halfTrack);

    setLeftVelocity(leftVelocity);
    setRightVelocity(rightVelocity);
}

void setupSteppers() {
    pinMode(LEFT_STEP_PIN, OUTPUT);
    pinMode(LEFT_DIR_PIN, OUTPUT);
    pinMode(LEFT_EN_PIN, OUTPUT);

    pinMode(RIGHT_STEP_PIN, OUTPUT);
    pinMode(RIGHT_DIR_PIN, OUTPUT);
    pinMode(RIGHT_EN_PIN, OUTPUT);

    digitalWrite(LEFT_STEP_PIN, LOW);
    digitalWrite(RIGHT_STEP_PIN, LOW);

    leftTimer = timerBegin(1000000);
    rightTimer = timerBegin(1000000);

    timerAttachInterrupt(leftTimer, &onLeftTimer);
    timerAttachInterrupt(rightTimer, &onRightTimer);

    timerAlarm(leftTimer, INITIAL_TIMER_PERIOD_US, true, 0);
    timerAlarm(rightTimer, INITIAL_TIMER_PERIOD_US, true, 0);

    timerStop(leftTimer);
    timerStop(rightTimer);

    disableMotors();
}


// ============================================================================
// IMU FUNCTIONS
// ============================================================================

bool writeImuRegister(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(MPU6050_ADDRESS);
    Wire.write(reg);
    Wire.write(value);

    return Wire.endTransmission() == 0;
}

bool readGyroRaw(int16_t& raw) {
    Wire.beginTransmission(MPU6050_ADDRESS);
    Wire.write(REG_GYRO_ZOUT_H);

    if (Wire.endTransmission(false) != 0) {
        return false;
    }

    if (Wire.requestFrom(MPU6050_ADDRESS, static_cast<uint8_t>(2)) != 2) {
        return false;
    }

    raw = static_cast<int16_t>(
        (Wire.read() << 8) | Wire.read()
    );

    return true;
}

bool setupImu() {
    Wire.begin();

    if (!writeImuRegister(REG_PWR_MGMT_1, 0x00)) {
        return false;
    }

    delay(100);

    if (!writeImuRegister(REG_GYRO_CONFIG, 0x00)) {
        return false;
    }

    delay(50);

    lastImuUpdateMicros = micros();

    return true;
}

bool calibrateImu() {
    int64_t sum = 0;

    for (uint16_t i = 0; i < IMU_CALIBRATION_SAMPLES; ++i) {
        int16_t raw;

        if (!readGyroRaw(raw)) {
            imuCalibrated = false;
            return false;
        }

        sum += raw;

        delay(IMU_CALIBRATION_DELAY_MS);
    }

    const float averageRaw =
        static_cast<float>(sum) /
        static_cast<float>(IMU_CALIBRATION_SAMPLES);

    gyroBiasDegPerSec = averageRaw / GYRO_SCALE_250DPS;

    imuHeadingDeg = 0.0f;
    gyroRateDegPerSec = 0.0f;
    rawGyroZ = 0;
    lastImuUpdateMicros = micros();

    imuCalibrated = true;

    return true;
}

void updateImu() {
    if (!imuCalibrated) {
        return;
    }

    const uint32_t now = micros();

    const float dt =
        static_cast<float>(now - lastImuUpdateMicros) / 1000000.0f;

    lastImuUpdateMicros = now;

    if (dt <= 0.0f || dt > 0.1f) {
        return;
    }

    int16_t raw;

    if (!readGyroRaw(raw)) {
        return;
    }

    rawGyroZ = raw;

    gyroRateDegPerSec =
        (static_cast<float>(raw) / GYRO_SCALE_250DPS) -
        gyroBiasDegPerSec;

    if (fabsf(gyroRateDegPerSec) < GYRO_DEADBAND_DPS) {
        gyroRateDegPerSec = 0.0f;
    }

    imuHeadingDeg += gyroRateDegPerSec * dt;
    imuHeadingDeg = normalizeAngle(imuHeadingDeg);
}

void resetHeading() {
    imuHeadingDeg = 0.0f;
    gyroRateDegPerSec = 0.0f;
    lastImuUpdateMicros = micros();
}


// ============================================================================
// MOTION CONTROL
// ============================================================================

void stopMotion() {
    commandVelocityMps = 0.0f;
    commandDesiredHeadingDeg = 0.0f;

    commandXLateralErrorM = 0.0f;
    commandYLateralErrorM = 0.0f;

    rampedVelocityMps = 0.0f;
    motionMode = MODE_STOP;

    baseDesiredHeadingDeg = 0.0f;

    initialLateralHeadingBiasDeg = 0.0f;
    activeLateralHeadingBiasDeg = 0.0f;
    activeTargetHeadingDeg = 0.0f;

    correctionDurationMs = 0.0f;
    correctionStartTimeMs = millis();

    headingErrorDeg = 0.0f;
    angularVelocityRadS = 0.0f;

    correctionStartLeftPulseCount = 0;
    correctionStartRightPulseCount = 0;
    correctionDistanceTargetM = 0.0f;
    correctionDistanceTravelledM = 0.0f;

    stopMotors();
}

void beginMotion() {
    stopMotion();

    lastCommandTimeMs = millis();
    lastMotionUpdateMicros = micros();
}

float computeApproachVelocity(float requestedVelocityMps) {
    const float y = commandYLateralErrorM;
    constexpr float Y_NEAR_M = 0.010f;
    constexpr float Y_FAR_M = 0.080f;
    constexpr float MIN_APPROACH_VELOCITY_MPS = 0.018f;

    float ratio = (y - Y_NEAR_M) / (Y_FAR_M - Y_NEAR_M);
    ratio = clampFloat(ratio, 0.0f, 1.0f);

    float velocity =
        MIN_APPROACH_VELOCITY_MPS +
        ratio * (requestedVelocityMps - MIN_APPROACH_VELOCITY_MPS);

    return limitVelocity(velocity);
}

void setVelocityCommand(
    float velocityMps,
    float desiredHeadingDeg,
    float lateralErrorM
) {
    motionMode = MODE_NORMAL;

    commandVelocityMps = limitVelocity(velocityMps);
    commandDesiredHeadingDeg = normalizeAngle(desiredHeadingDeg);

    commandXLateralErrorM = lateralErrorM;
    commandYLateralErrorM = 0.0f;

    if (fabsf(commandVelocityMps) < 0.001f) {
        stopMotion();
        lastCommandTimeMs = millis();
        return;
    }

    baseDesiredHeadingDeg = commandDesiredHeadingDeg;

    float correctionDistanceM =
        TAG_SPACING_M * LATERAL_CORRECTION_DISTANCE_RATIO;

    if (correctionDistanceM < 0.05f) {
        correctionDistanceM = 0.05f;
    }

    initialLateralHeadingBiasDeg =
        atan2f(
            LATERAL_GAIN * commandXLateralErrorM,
            correctionDistanceM
        ) * RAD_TO_DEG_F;

    initialLateralHeadingBiasDeg =
        clampFloat(
            initialLateralHeadingBiasDeg,
            -MAX_LATERAL_HEADING_BIAS_DEG,
            MAX_LATERAL_HEADING_BIAS_DEG
        );

    activeLateralHeadingBiasDeg = initialLateralHeadingBiasDeg;

    activeTargetHeadingDeg =
        normalizeAngle(baseDesiredHeadingDeg + activeLateralHeadingBiasDeg);

    noInterrupts();

    correctionStartLeftPulseCount = leftStepPulseCount;
    correctionStartRightPulseCount = rightStepPulseCount;

    interrupts();

    correctionDistanceTargetM = correctionDistanceM;
    correctionDistanceTravelledM = 0.0f;

    correctionDurationMs = 0.0f;
    correctionStartTimeMs = millis();
    lastCommandTimeMs = millis();
}

void setApproachCommand(
    float velocityMps,
    float desiredHeadingDeg,
    float xLateralErrorM,
    float yLateralErrorM
) {
    motionMode = MODE_APPROACH;

    commandVelocityMps = limitVelocity(velocityMps);
    commandDesiredHeadingDeg = normalizeAngle(desiredHeadingDeg);

    commandXLateralErrorM = xLateralErrorM;
    commandYLateralErrorM = yLateralErrorM;

    if (fabsf(commandVelocityMps) < 0.001f) {
        stopMotion();
        lastCommandTimeMs = millis();
        return;
    }

    baseDesiredHeadingDeg = commandDesiredHeadingDeg;

    float correctionDistanceM =
        TAG_SPACING_M * LATERAL_CORRECTION_DISTANCE_RATIO;

    if (correctionDistanceM < 0.05f) {
        correctionDistanceM = 0.05f;
    }

    initialLateralHeadingBiasDeg =
        atan2f(
            LATERAL_GAIN * commandXLateralErrorM,
            correctionDistanceM
        ) * RAD_TO_DEG_F;

    initialLateralHeadingBiasDeg =
        clampFloat(
            initialLateralHeadingBiasDeg,
            -MAX_LATERAL_HEADING_BIAS_DEG,
            MAX_LATERAL_HEADING_BIAS_DEG
        );

    activeLateralHeadingBiasDeg = initialLateralHeadingBiasDeg;

    activeTargetHeadingDeg =
        normalizeAngle(baseDesiredHeadingDeg + activeLateralHeadingBiasDeg);

    // In approach mode, do not fade correction.
    // Python keeps sending latest APP x/y values from latest camera frame.
    correctionDurationMs = 0.0f;
    correctionStartTimeMs = millis();
    lastCommandTimeMs = millis();
}


// ============================================================================
// TURNING
// ============================================================================

void startTurn(float targetHeadingDeg) {
    stopMotion();

    turnTargetHeadingDeg = normalizeAngle(targetHeadingDeg);
    turnStartTimeMs = millis();
    turnActive = true;
}

void stopTurn() {
    turnActive = false;
    turnTargetHeadingDeg = 0.0f;
    turnStartTimeMs = 0;
}

void updateTurn() {
    if (!turnActive) {
        return;
    }

    if (!motorsEnabled) {
        turnActive = false;
        stopMotion();
        Serial.println("ERR TURN EN");
        return;
    }

    if (!imuCalibrated) {
        turnActive = false;
        stopMotion();
        Serial.println("ERR TURN IMU");
        return;
    }

    float errorDeg = normalizeAngle(turnTargetHeadingDeg - imuHeadingDeg);

    if (fabsf(errorDeg) <= TURN_TOLERANCE_DEG) {
        turnActive = false;
        stopMotion();
        Serial.println("TURN_DONE");
        return;
    }

    if ((millis() - turnStartTimeMs) > TURN_TIMEOUT_MS) {
        turnActive = false;
        stopMotion();
        Serial.println("FAULT TURN_TIMEOUT");
        return;
    }

    float wz = TURN_KP * errorDeg * DEG_TO_RAD_F;
    wz = clampFloat(wz, -TURN_MAX_WZ, TURN_MAX_WZ);

    if (fabsf(wz) < TURN_MIN_WZ) {
        wz = (wz >= 0.0f) ? TURN_MIN_WZ : -TURN_MIN_WZ;
    }

    angularVelocityRadS = TURN_DIRECTION * wz;
    headingErrorDeg = errorDeg;
    activeTargetHeadingDeg = turnTargetHeadingDeg;

    setMotion(0.0f, angularVelocityRadS);
}

void updateMotion() {
    if (turnActive) {
        return;
    }

    const uint32_t now = micros();
    const uint32_t elapsedUs = now - lastMotionUpdateMicros;

    if (elapsedUs < CONTROL_PERIOD_US) {
        return;
    }

    lastMotionUpdateMicros = now;

    const float controlDt =
        static_cast<float>(elapsedUs) / 1000000.0f;

    if (fabsf(commandVelocityMps) < 0.001f) {
        headingErrorDeg = 0.0f;
        angularVelocityRadS = 0.0f;

        activeLateralHeadingBiasDeg = 0.0f;
        activeTargetHeadingDeg = 0.0f;

        rampedVelocityMps = 0.0f;

        stopMotors();

        return;
    }

    if (motionMode == MODE_APPROACH) {
        // Approach mode:
        // keep latest x-lateral correction active.
        activeLateralHeadingBiasDeg = initialLateralHeadingBiasDeg;
    } else {
        // Normal mode:
        // fade lateral correction by travelled pulse distance.
        correctionDistanceTravelledM = getCorrectionTravelledDistanceM();

        float progress = 1.0f;

        if (correctionDistanceTargetM > 0.001f) {
            progress =
                correctionDistanceTravelledM /
                correctionDistanceTargetM;

            progress = clampFloat(progress, 0.0f, 1.0f);
        }

        activeLateralHeadingBiasDeg =
            initialLateralHeadingBiasDeg * (1.0f - progress);
    }

    activeTargetHeadingDeg =
        normalizeAngle(baseDesiredHeadingDeg + activeLateralHeadingBiasDeg);

    headingErrorDeg =
        normalizeAngle(activeTargetHeadingDeg - imuHeadingDeg);

    const float headingCorrection =
        HEADING_KP * headingErrorDeg;

    angularVelocityRadS =
        STEERING_DIRECTION *
        limitAngularVelocity(headingCorrection * DEG_TO_RAD_F);

    float targetLinearVelocity = commandVelocityMps;

    if (motionMode == MODE_APPROACH) {
        targetLinearVelocity = computeApproachVelocity(commandVelocityMps);
    }

    const float maxVelocityChange = LINEAR_ACCEL_MPS2 * controlDt;

    rampedVelocityMps = rampToward(
        rampedVelocityMps,
        targetLinearVelocity,
        maxVelocityChange
    );

    setMotion(rampedVelocityMps, angularVelocityRadS);
}


// ============================================================================
// SERIAL PROTOCOL
// ============================================================================

void printStatus() {
    Serial.print("STATUS ");

    Serial.print("EN=");
    Serial.print(motorsEnabled);

    Serial.print(" CAL=");
    Serial.print(imuCalibrated);

    Serial.print(" TURN=");
    Serial.print(turnActive);

    Serial.print(" MODE=");
    Serial.print(static_cast<int>(motionMode));

    Serial.print(" HDG=");
    Serial.print(imuHeadingDeg, 2);

    Serial.print(" GYRO=");
    Serial.print(gyroRateDegPerSec, 2);

    Serial.print(" DES=");
    Serial.print(commandDesiredHeadingDeg, 2);

    Serial.print(" ACTIVE=");
    Serial.print(activeTargetHeadingDeg, 2);

    Serial.print(" BIAS=");
    Serial.print(activeLateralHeadingBiasDeg, 2);

    Serial.print(" ERR=");
    Serial.print(headingErrorDeg, 2);

    Serial.print(" WZ=");
    Serial.print(angularVelocityRadS, 4);

    Serial.print(" VEL=");
    Serial.print(commandVelocityMps, 3);

    Serial.print(" RV=");
    Serial.print(rampedVelocityMps, 4);

    Serial.print(" XLAT=");
    Serial.print(commandXLateralErrorM, 4);

    Serial.print(" YLAT=");
    Serial.print(commandYLateralErrorM, 4);

    Serial.print(" CDIST=");
    Serial.print(correctionDistanceTravelledM, 4);

    Serial.print(" CTARGET=");
    Serial.print(correctionDistanceTargetM, 4);

    Serial.print(" LVM=");
    Serial.print(leftVelocityMps, 4);

    Serial.print(" RVM=");
    Serial.print(rightVelocityMps, 4);

    Serial.print(" LPPS=");
    Serial.print(leftStepRatePps, 1);

    Serial.print(" RPPS=");
    Serial.println(rightStepRatePps, 1);
}

void processCommand(char* line) {
    char* cmd = strtok(line, " ");

    if (cmd == nullptr) {
        return;
    }

    if (!strcasecmp(cmd, "PING")) {
        Serial.println("ACK");
        return;
    }

    if (!strcasecmp(cmd, "EN")) {
        enableMotors();
        Serial.println("ACK");
        return;
    }

    if (!strcasecmp(cmd, "DIS")) {
        stopTurn();
        stopMotion();
        disableMotors();
        commandActive = false;
        Serial.println("ACK");
        return;
    }

    if (!strcasecmp(cmd, "STOP")) {
        stopTurn();
        stopMotion();
        commandActive = false;
        Serial.println("ACK");
        return;
    }

    if (!strcasecmp(cmd, "CAL")) {
        stopTurn();
        stopMotion();
        commandActive = false;

        calibrating = true;
        Serial.println("CAL START");

        if (calibrateImu()) {
            calibrating = false;
            Serial.println("ACK");
        } else {
            calibrating = false;
            Serial.println("ERR CAL");
        }

        return;
    }

    if (!strcasecmp(cmd, "ZERO")) {
        resetHeading();
        Serial.println("ACK");
        return;
    }

    if (!strcasecmp(cmd, "STATUS")) {
        printStatus();
        return;
    }

    if (!strcasecmp(cmd, "VEL")) {
        stopTurn();

        char* velocityString = strtok(nullptr, " ");
        char* headingString = strtok(nullptr, " ");
        char* lateralString = strtok(nullptr, " ");

        if (
            velocityString == nullptr ||
            headingString == nullptr ||
            lateralString == nullptr
        ) {
            Serial.println("ERR VEL");
            return;
        }

        const float velocityMps = atof(velocityString);
        const float desiredHeadingDeg = atof(headingString);
        const float lateralErrorM = atof(lateralString);

        setVelocityCommand(
            velocityMps,
            desiredHeadingDeg,
            lateralErrorM
        );

        lastCommandTimeMs = millis();
        commandActive = fabsf(commandVelocityMps) > 0.001f;

        Serial.println("ACK");
        return;
    }

    if (!strcasecmp(cmd, "APP")) {
        stopTurn();

        char* velocityString = strtok(nullptr, " ");
        char* headingString = strtok(nullptr, " ");
        char* xLateralString = strtok(nullptr, " ");
        char* yLateralString = strtok(nullptr, " ");

        if (
            velocityString == nullptr ||
            headingString == nullptr ||
            xLateralString == nullptr ||
            yLateralString == nullptr
        ) {
            Serial.println("ERR APP");
            return;
        }

        const float velocityMps = atof(velocityString);
        const float desiredHeadingDeg = atof(headingString);
        const float xLateralErrorM = atof(xLateralString);
        const float yLateralErrorM = atof(yLateralString);

        setApproachCommand(
            velocityMps,
            desiredHeadingDeg,
            xLateralErrorM,
            yLateralErrorM
        );

        lastCommandTimeMs = millis();
        commandActive = fabsf(commandVelocityMps) > 0.001f;

        Serial.println("ACK");
        return;
    }

    if (!strcasecmp(cmd, "TURN")) {
        char* headingString = strtok(nullptr, " ");

        if (headingString == nullptr) {
            Serial.println("ERR TURN");
            return;
        }

        const float targetHeadingDeg = atof(headingString);

        startTurn(targetHeadingDeg);

        commandActive = false;

        Serial.println("ACK");
        return;
    }

    Serial.println("ERR CMD");
}

void readCommand() {
    while (Serial.available()) {
        const char c = static_cast<char>(Serial.read());

        if (c == '\r') {
            continue;
        }

        if (c == '\n') {
            commandBuffer[commandIndex] = '\0';

            if (commandIndex > 0) {
                processCommand(commandBuffer);
            }

            commandIndex = 0;

            return;
        }

        if (commandIndex < sizeof(commandBuffer) - 1) {
            commandBuffer[commandIndex++] = c;
        }
    }
}

void updateSerial() {
    readCommand();

    if (
        commandActive &&
        !turnActive &&
        ((millis() - lastCommandTimeMs) > COMMAND_TIMEOUT_MS)
    ) {
        stopMotion();
        commandActive = false;
        Serial.println("FAULT TIMEOUT");
    }
}

void sendStatusIfDue() {
    if (calibrating) {
        return;
    }

    if ((millis() - lastStatusTimeMs) < STATUS_PERIOD_MS) {
        return;
    }

    lastStatusTimeMs = millis();

    printStatus();
}


// setup and main loop

void setup() {
    Serial.begin(115200);
    delay(500);

    setupSteppers();
    disableMotors();

    if (!setupImu()) {
        Serial.println("ERR IMU");
    } else {
        Serial.println("IMU OK");
    }

    beginMotion();

    lastStatusTimeMs = millis();
    lastCommandTimeMs = millis();

    Serial.println("AGV Ready");
}

void loop() {
    updateSerial();
    updateImu();
    updateTurn();
    updateMotion();
    sendStatusIfDue();
}