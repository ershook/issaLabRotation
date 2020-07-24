#include <SoftwareSerial.h>

SoftwareSerial mySerial(8,9);
unsigned long time;

byte val = 0; // variable to store the data from the serial port

void setup() {
Serial.begin(9600); // connect to the serial port
mySerial.begin(9600); //conect to the software serial port
}

void loop () {


while(mySerial.available() > 0) {
  // look at buffer... 
  
  //Serial.print("Time: ");
  //time = millis();

  //Serial.println(time); //prints time since program started
  val =  mySerial.read();
  Serial.write(val);
  //Serial.println();
  }
}
