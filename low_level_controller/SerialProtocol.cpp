#include "SerialProtocol.h"

SerialProtocol::SerialProtocol(
    Stepper& stepper,
    Imu& imu,
    Motion& motion)
    : stepper(stepper),
      imu(imu),
      motion(motion),
      commandIndex(0),
      lastCommandTime(0),
      lastStatusTime(0),
      commandActive(false) {

    commandBuffer[0] = '\0';
}

void SerialProtocol::begin() {
    Serial.begin(Config::SERIAL_BAUD_RATE);

    delay(500);

    lastCommandTime = millis();
    lastStatusTime = millis();

    Serial.println("AGV Ready");
}

void SerialProtocol::update() {
    readCommand();

    if (commandActive && (millis() - lastCommandTime > Config::COMMAND_TIMEOUT_MS)) {

        motion.stop();

        commandActive = false;

        Serial.println("FAULT TIMEOUT");
    }
}

void SerialProtocol::sendStatusIfDue() {
    if ((millis() - lastStatusTime) < Config::STATUS_PERIOD_MS) {
        return;
    }

    lastStatusTime = millis();

    printStatus();
}

void SerialProtocol::readCommand() {
    while (Serial.available()) {

        char c = static_cast<char>(Serial.read());

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

void SerialProtocol::processCommand(char* line) {

    char* cmd = strtok(line, " ");

    if (cmd == nullptr) {
        return;
    }

    if (!strcasecmp(cmd, "PING")) {

        Serial.println("ACK");

        return;
    }

    if (!strcasecmp(cmd, "EN")) {

        stepper.enable();

        Serial.println("ACK");

        return;
    }

    if (!strcasecmp(cmd, "DIS")) {

        motion.stop();

        stepper.disable();

        commandActive = false;

        Serial.println("ACK");

        return;
    }

    if (!strcasecmp(cmd, "STOP")) {

        motion.stop();

        commandActive = false;

        Serial.println("ACK");

        return;
    }

    if (!strcasecmp(cmd, "CAL")) {

        motion.stop();

        commandActive = false;

        if (imu.calibrate()) {
            Serial.println("ACK");
        } else {
            Serial.println("ERR");
        }

        return;
    }

    if (!strcasecmp(cmd, "ZERO")) {

        imu.resetHeading();

        Serial.println("ACK");

        return;
    }

    if (!strcasecmp(cmd, "STATUS")) {

        printStatus();

        return;
    }

    if (!strcasecmp(cmd, "VEL")) {

        handleVelocityCommand();

        return;
    }

    Serial.println("ERR");
}

void SerialProtocol::handleVelocityCommand() {

    char* velocityString = strtok(nullptr, " ");
    char* headingString = strtok(nullptr, " ");
    char* lateralString = strtok(nullptr, " ");

    if (velocityString == nullptr || headingString == nullptr || lateralString == nullptr) {

        Serial.println("ERR");

        return;
    }

    NavigationCommand command;

    command.velocityMps = atof(velocityString);

    command.desiredHeadingDeg = atof(headingString);

    command.lateralErrorM = atof(lateralString);

    motion.setNavigationCommand(command);

    lastCommandTime = millis();

    commandActive = fabs(command.velocityMps) > 0.0f;

    Serial.println("ACK");
}

void SerialProtocol::printStatus() {

    NavigationCommand command = motion.getNavigationCommand();

    Serial.print("STATUS ");

    Serial.print("EN=");
    Serial.print(stepper.isEnabled());

    Serial.print(" CAL=");
    Serial.print(imu.isCalibrated());

    Serial.print(" HDG=");
    Serial.print(imu.getHeading(), 2);

    Serial.print(" GYRO=");
    Serial.print(imu.getGyroRate(), 2);

    Serial.print(" DES=");
    Serial.print(command.desiredHeadingDeg, 2);

    Serial.print(" ACTIVE=");
    Serial.print(motion.getActiveTargetHeading(), 2);

    Serial.print(" BIAS=");
    Serial.print(motion.getActiveLateralBias(), 2);

    Serial.print(" ERR=");
    Serial.print(motion.getHeadingError(), 2);

    Serial.print(" WZ=");
    Serial.print(motion.getAngularVelocity(), 4);

    Serial.print(" VEL=");
    Serial.print(command.velocityMps, 3);

    Serial.print(" LAT=");
    Serial.println(command.lateralErrorM, 4);
}

void SerialProtocol::printHelp() {

    Serial.println();

    Serial.println("PING");
    Serial.println("EN");
    Serial.println("DIS");
    Serial.println("STOP");
    Serial.println("CAL");
    Serial.println("ZERO");
    Serial.println("STATUS");
    Serial.println("VEL velocity desiredHeading lateralError");

    Serial.println();
}