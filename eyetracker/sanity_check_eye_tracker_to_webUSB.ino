#include <SoftwareSerial.h>
#include <WebUSB.h>

WebUSB WebUSBSerial(1, "webusb.github.io/arduino/demos");
#define Serial WebUSBSerial


SoftwareSerial mySerial(8, 9);
int RFIDResetPin = 11; 

byte val = 0; // variable to store the data from the serial port

void setup() {
Serial.begin(9600); // connect to the serial port
mySerial.begin(57600);

}

void loop () {


while(mySerial.available() > 0) {
digitalWrite(LED_BUILTIN, HIGH);
// look at buffer... 
val =  mySerial.read();

Serial.write(val);
//Serial.print(val);
Serial.flush();
}
}
