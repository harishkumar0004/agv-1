#pragma once

#include <Arduino.h>

namespace Config {

//==================================================
// Serial
//==================================================

constexpr uint32_t SERIAL_BAUD_RATE = 115200;

//==================================================
// Stepper Driver Pins
//==================================================

constexpr uint8_t LEFT_STEP_PIN = 16;
constexpr uint8_t LEFT_DIR_PIN = 26;
constexpr uint8_t LEFT_EN_PIN = 25;

constexpr uint8_t RIGHT_STEP_PIN = 4;
constexpr uint8_t RIGHT_DIR_PIN = 13;
constexpr uint8_t RIGHT_EN_PIN = 14;

constexpr uint8_t DRIVER_ENABLE_LEVEL = LOW;
constexpr uint8_t DRIVER_DISABLE_LEVEL = HIGH;

//==================================================
// ESP32 Timers
//==================================================

constexpr uint8_t LEFT_TIMER_ID = 0;
constexpr uint8_t RIGHT_TIMER_ID = 1;

constexpr uint16_t TIMER_PRESCALER = 80;
constexpr uint32_t INITIAL_TIMER_PERIOD_US = 1000;

//==================================================
// Robot Geometry
//==================================================
// constexpr float PI = 3.14159265358979323846f;
constexpr float WHEEL_DIAMETER_M = 0.117f;
constexpr uint32_t STEPS_PER_WHEEL_REV = 20000;
constexpr float WHEEL_CIRCUMFERENCE_M = PI * WHEEL_DIAMETER_M;
constexpr float STEPS_PER_METER = STEPS_PER_WHEEL_REV / WHEEL_CIRCUMFERENCE_M;
constexpr float TRACK_WIDTH_M = 0.324;

//==================================================
// Motion Limits
//==================================================

constexpr float MAX_LINEAR_VELOCITY_MPS = 0.10f;
constexpr float MAX_LINEAR_ACCELERATION_MPS2 = 0.30f;
constexpr float MAX_STEP_RATE = 10000.0f;
constexpr float MIN_STEP_RATE = 1.0f;

//==================================================
// MPU6050
//==================================================
constexpr uint8_t MPU6050_ADDRESS = 0x68;
constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;
constexpr uint8_t REG_GYRO_CONFIG = 0x1B;
constexpr uint8_t REG_GYRO_ZOUT_H = 0x47;

constexpr float GYRO_SCALE_250DPS = 131.0f;

constexpr uint16_t IMU_CALIBRATION_SAMPLES = 1000;
constexpr uint16_t IMU_CALIBRATION_DELAY_MS = 2;

// Visual heading correction.
constexpr float HEADING_CORRECTION_ALPHA = 0.03f;

//==================================================
// Heading
//==================================================
constexpr float FULL_CIRCLE_DEG = 360.0f;
constexpr float HEADING_MAX_DEG = 180.0f;
constexpr float HEADING_MIN_DEG = -180.0f;

//==================================================
// Controller
//==================================================

constexpr float HEADING_KP = 0.2f;
constexpr float HEADING_KI = 0.0f;
constexpr float HEADING_KD = 0.0f;

constexpr float LATERAL_KP = 0.01f;

constexpr float TAG_SPACING_M = 0.50f; // tag spacing in meters.

constexpr float LATERAL_CORRECTION_DISTANCE_RATIO = 0.70f; // 70percent corrrection happens in tag spacing.

constexpr float MAX_LATERAL_HEADING_BIAS_DEG = 10.0f; // clamp lateral-to-heading correction.

constexpr float MAX_STEERING = 0.05f;

constexpr float STEERING_DIRECTION = 1.0f;

//==================================================
// Timing
//==================================================
constexpr uint32_t CONTROL_PERIOD_US = 2500;
constexpr uint32_t COMMAND_TIMEOUT_MS = 30000;
constexpr uint32_t STATUS_PERIOD_MS = 250;

}