from time import sleep
import serial

"""
This library contains the PowerSupply object and other functions required to control the power supplies.

"""


class PowerSupply:

    def __init__(self, port, baud_rate) -> None:
        self.ser = serial.Serial(port, baud_rate)
        self.offset = 2.5797
        self.conversion = 49.917
        self.currentB = 0
        self.getConnectionStatus()

    def getID(self) -> None:
        # Prints info about serial connection

        print('Port: ' + str(self.ser.name))
        print("Connection open: " + str(self.ser.isOpen()))
        self.ser.write(b'*IDN?\r')
        response = self.read_to_r()
        print("ID: " + str(response))

    def getConnectionStatus(self) -> None:
        # Prints info about serial connection

        if self.ser.isOpen():
            print("Power supply connected on port " + str(self.ser.name))

        else:
            print("Power supply not connected.")

    def setOffsetConversion(self, offset, conversion) -> None:
        self.offset = offset
        self.conversion = conversion

    def setCurrent(self, i: float, give_additional_info=False) -> None:
        # Sets output current to given value
        # Also sets the output states to 0 if current is set to 0, to 1 if current set to any other value
        # i : Output current [A]

        # For Kepco power supplies uncomment the following line that sets the operation mode to "Current mode"
        # self.ser.write(bytes('FUNC:MODE {CURR}', 'utf-8'))

        maxCurrent = 3.6

        self.currentB = 0

        if abs(i) > maxCurrent:
            print(f'abs(i) A exceeds max current of ' + str(maxCurrent) + ' A')
            return
        # elif abs(i) > 2:
        #     self.logger.warning('Using current greater than 0.5 A, make sure the electromagnet is properly cooled')

        # For F2031 power supplies uncomment the following line
        command = 'CUR {current:+}\r'.format(current=i)

        # The following command selects the operating current range for the Kepco power supply. The possible options are '1' and '4', where '1' means full scale and '4' means 1/4 of the full scale
        # self.ser.write(bytes('[SOUR:]CURR[:LEV]:RANG 4', 'utf-'))

        # Command that decides the output current for the Kepco power supply
        # command = '[SOUR:]CURR[:LEV][:IMM][:AMP] {current:+}\r'.format(current=i)
        if give_additional_info:
            print('Query:', command)

        self.ser.write(bytes(command, 'utf-8'))  # query to set current

        # Uncomment these lines if you are using the F2031 power supplies
        response = self.read_to_r()

        if response != 'CMLT\r':
            print("Unexpected response in function setCurrent, " + response)

        if i == 0:
            self.setOutputState(0)
        else:
            self.setOutputState(1)

        return

    def setField(self, B: float, give_additional_info=False) -> None:
        # Sets output field to given value [mT]
        # Also sets the output states to 0 if field is set to 0, to 1 if field set to any other value
        # B : Output field [mT]

        if (B!= self.currentB):

            I = (B - self.offset) / self.conversion  # Calculate current corresponding to desired field

            self.setCurrent(I, give_additional_info)

            self.currentB = B

    def setOutputState(self, state: int) -> None:
        # Sets state to:
        # 1: Output state
        # 0: High impedance state

        # For Kepco power supply uncomment the following line
        # self.ser.write(bytes('OUTP[:STAT] {state}\r'.format(state=state), 'utf-8'))

        # For F2031 power supplies uncomment the following lines
        self.ser.write(bytes('OUT {state}\r'.format(state=state), 'utf-8'))

        response = self.read_to_r()
        if response != 'CMLT\r':
            print("Unexpected response in function setOutputState, " + response)
            response = self.read_to_r()
            print("Unexpected response in function setOutputState, " + response)

        return

    # Only used for F2031 power supplies
    def setRampRate(self, rate: float) -> None:
        # Sets ramp rate [A/s]
        # Rate should be in range the range 0.01 A/s < range < 2 A/s

        if rate < 0.01:
            rate = 0.01
            print("Using rate smaller than 0.01 A/s, using 0.01 A/s instead")

        if rate > 2:
            rate = 2
            print('Using rate greater than 2 A/s, using 2 A/s instead')

        command = 'RATE {rate}\r'.format(rate=rate)
        print('Query:', command)

        self.ser.write(bytes(command, 'utf-8'))  # query to set current

        response = self.read_to_r()
        if response != 'CMLT\r':
            print("Unexpected response in function setRampRate, " + response)

        return

    # TODO used above, replace with better method
    def read_to_r(self) -> str:
        # readline does not work, because termination character is \r instead of default \n

        ch = ''
        line = []
        while ch != b'\r':
            ch = self.ser.read()
            line.append(ch.decode('utf-8'))
        return ''.join(line)

    def closeConnection(self) -> None:
        self.ser.close()

    # Only used for F2031 power supplies
    def demag_sweep(self) -> None:
        demag_sweep = [3, -1.5, 0.75, -0.375, 0.1875, -0.09375, 0.045, -0.02, 0.01, -0.005, 0.002, -0.001, 0.0005]
        print("Executing demagnetizing sweep...")

        for current in demag_sweep:
            self.setCurrent(current)
            sleep(0.5)

        self.setCurrent(0)
        print("Completed demagnetizing sweep.\n")

    # Only used for F2031 power supplies
    def setTriggers(self, val, give_additional_info=False) -> None:
        command = 'SWTRIG n{val}'
        if give_additional_info:
            print('Query:', command)

        self.ser.write(bytes(command, 'utf-8'))

        command = 'NTRIG n{val}'
        if give_additional_info:
            print('Query:', command)

        self.ser.write(bytes(command, 'utf-8'))  # query to set current
