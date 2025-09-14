/* This code is sued for starting continous ranging on SRF02 sensor
 *
 * (c) Red Pitaya  http://www.redpitaya.com
 *
 * This part of code is written in C programming language.
 * Please visit http://en.wikipedia.org/wiki/C_(programming_language)
 * for more details on the language used herein.
 * it is in /usr/RedPitaya/Examples/C
 * Possible decimations:
 * RP_DEC_1, RP_DEC_2, RP_DEC_4, RP_DEC_8, RP_DEC_16 , RP_DEC_32 , RP_DEC_64 ,
 * RP_DEC_128, RP_DEC_256, RP_DEC_512, RP_DEC_1024, RP_DEC_2048, RP_DEC_4096, RP_DEC_8192,
 * RP_DEC_16384, RP_DEC_32768, RP_DEC_65536
 */

// ***************************************************** //
// *************** Libray Integration ****************** //
// ***************************************************** //
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <linux/ioctl.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>
#include <sys/socket.h>


#include <stdint.h>
#include "rp.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <syslog.h>
#include <sys/time.h>

// ***************************************************** //
// ********* Static Global Variable Declaration ******** //
// ***************************************************** //

//Defining IsquareC protocol addresses (Start8bitSlaveAddr-Ack8bitInternalRegisterAddr-Ack8bitData-AckStop)
#define I2C_SLAVE_FORCE                 0x0706      // This is for the Forced Slave bus address
#define I2C_SLAVE                       0x0703      // This is for the Slave bus address
#define I2C_FUNCS                       0x0705      // This is for the I square C functions defining address
#define I2C_RDWR                        0x0707      // This is for I square C data Read and Write address

//Define Sensor frequency and other properties
#define V_SONIC_WAVE                    (float)(343.2)  // [m/s] Schallgeschwindigkeit in Luft
#define DECIMATION_FACTOR                       RP_DEC_64       // This is the descimation factor or number of points channel 1 will take in 1 second

#define ADC_MAX_SAMPLE_FREQUENCY        125000000                                               // [Hz] --> 125 MHz
#define ADC_SAMPLE_DECIMATION           64                                                      // [-]
#define ADC_SAMPLE_FREQUENCY            ( ADC_MAX_SAMPLE_FREQUENCY / ADC_SAMPLE_DECIMATION )
#define ADC_SAMPLE_TIME                 8               // [ns]
#define ADC_SAMPLE_TIME_NS              (uint32_t)( ADC_SAMPLE_DECIMATION * ADC_SAMPLE_TIME )   // [ns] --> 8*64=512 ns / sample
#define ADC_START_DELAY_US              (uint32_t)( 0.30 * 2 * 1e6 / V_SONIC_WAVE )     // [µs] --> 2 * 0,30 m / 343,2 m/s = 1.748 µs



//Define the whole data size to capture through the SONAR sensor and the sending data size through UDP
#define DATA_SIZE                       25000
#define READ_DATA_SIZE                  25000

//Initiate all the Redpitaya board led state (from LED 7-9 is used by system)
#define LEDx_INIT()                     rp_DpinSetDirection( RP_LED0, RP_OUT ), \
                                        rp_DpinSetDirection( RP_LED1, RP_OUT ), \
                                        rp_DpinSetDirection( RP_LED2, RP_OUT ), \
                                        rp_DpinSetDirection( RP_LED3, RP_OUT ), \
                                        rp_DpinSetDirection( RP_LED4, RP_OUT ), \
                                                            rp_DpinSetDirection( RP_LED5, RP_OUT ), \
                                                            rp_DpinSetDirection( RP_LED6, RP_OUT ), \
                                        rp_DpinSetState( RP_LED0, RP_LOW ),     \
                                        rp_DpinSetState( RP_LED1, RP_LOW ),     \
                                        rp_DpinSetState( RP_LED2, RP_LOW ),     \
                                        rp_DpinSetState( RP_LED3, RP_LOW ),     \
                                        rp_DpinSetState( RP_LED4, RP_LOW ),     \
                                        rp_DpinSetState( RP_LED5, RP_LOW ),     \
                                                            rp_DpinSetState( RP_LED6, RP_LOW )

//Define LED
#define LED0_OFF                        rp_DpinSetState( RP_LED0, RP_LOW  )
#define LED1_OFF                        rp_DpinSetState( RP_LED1, RP_LOW  )
#define LED2_OFF                        rp_DpinSetState( RP_LED2, RP_LOW  )
#define LED3_OFF                        rp_DpinSetState( RP_LED3, RP_LOW  )
#define LED4_OFF                        rp_DpinSetState( RP_LED4, RP_LOW  )
#define LED5_OFF                        rp_DpinSetState( RP_LED5, RP_LOW  )
#define LED6_OFF                        rp_DpinSetState( RP_LED6, RP_LOW  )
#define LED0_ON                         rp_DpinSetState( RP_LED0, RP_HIGH )
#define LED1_ON                         rp_DpinSetState( RP_LED1, RP_HIGH )
#define LED2_ON                         rp_DpinSetState( RP_LED2, RP_HIGH )
#define LED3_ON                         rp_DpinSetState( RP_LED3, RP_HIGH )
#define LED4_ON                         rp_DpinSetState( RP_LED4, RP_HIGH )
#define LED5_ON                         rp_DpinSetState( RP_LED5, RP_HIGH )
#define LED6_ON                         rp_DpinSetState( RP_LED6, RP_HIGH )

//Define the IP address of the connected PC and the port to send data through UDP
#define UDP_PORT                        61231               // This is the connecting UDP port of the PC
#define SW_VERSION                      2.00
// #define LOG_ACTIVITIES                  1                   // This value is 1 while debugging on points and logging otherwise put into comment

// ***************************************************** //
// ************ Enumerator Type Definations ************ //
// ***************************************************** //

/* Type definatons for global struct of iic */
struct iic_s
{
    int         fd;
    int         address;
    char        buf[4];
    char        *fileName;
}iic = {
    .address    = 0x70,
    .fileName   = "/dev/i2c-0"
};

/* Typedefs for data structs: Header and data */
typedef struct
{
        float HeaderLength; //this define the header length for client packet reading (17*4)Bytes[C]
        float DataLength; //this defines the data length of the current sent packet in bytes (25000*2)Bytes [C]
        float Class;  //this defines what type of object is infront of the sensor [V]
        float DataType; //this has to be the data type currently only ADC [V]
        float X_Interval; //This is the ADC sample time in nano seconds [C]
    float Scaling; //There is no usage of this yet [C]
        float SampleFrequency; //sampling frequency of the ultrasonic signal [C]
        float ADCResolution; //Found from Redpitaya documentation sheet [C]
        float AmbTemperature; //Fixed temperature or coiming from a temperature sensor [C]
        float MeasDelay; //This is the ADC start Delay for only stopping capturing the sending pulse [C]
        float Distance; //this gives the sensor detected distance of the object calculated from reflection time of fly [V]
        float FFTWindowLength; //These fields are conducting with FFT Data only FFT window length [V]
        float FFTWindowOffsetIndex; //These fields are conducting with FFT Data only FFT window offset value [V]
        float SWVersion; //This is the constant software version [C]
        float TotalDataBlocks; //this gives how many datablocks per signal will be sent [C]
        float DataBlockNumber; //th6is gives the current sent data block number [V]
    float CurrentTime; //Previously It was Scaling as the scaling factor has no significance in ADC data it has been replaced with current time in the redpitaya os after the udp connection [V]
        uint16_t p_data[READ_DATA_SIZE]; //this holds the actual ADC data [V]
}Header_t;

/* Typedefs for data structure: Header and data */
typedef struct
{
        int         socket;
        socklen_t   addr_length;
        char        command;
    struct      sockaddr_in server_addr, client_addr;
    int32_t     parameter[2];
}udp_t;

/* Typedefs for run variable: either send only the header or send the data */
typedef enum
{
        RUN_NONE,
        RUN_ADC
}run_t;

// ***************************************************** //
// ************ Static Variable Declarations *********** //
// ***************************************************** //
static udp_t                udp={-1, -1};
static Header_t             Header_with_data;
static run_t                run;
//static const uint32_t         thread_pause_us = THREAD_PAUSE_US;
static float                            time_of_flight_us;
static float                            distance;
static volatile bool            run_thread = true;
static volatile float       current_time = 0; // Global time variable
pthread_mutex_t                         run_meas_mutex;
FILE                        *log_file;

// ***************************************************** //
// *************** Function Declarations *************** //
// ***************************************************** //
int         initiate_redpitaya_with_dma();
void        initiate_udp_connection();
void        initiate_iic();
int         acquire_data();
void        read_data(int16_t *data, int block_number);
void        release_resources();
static void parse_udp_message( char *msg_rx );
void        send_binary( uint8_t *data_in, uint16_t count );
void        iic_set_run( run_t run_in, bool param );
void        log_activities(char *message, int line);
int         re_initiate_dma_channel();
void        close_dma_channel();
void        measure_distance(float *distance_out);

void        *Server_thread();
void        *Measure_thread();
void        *Time_thread();


// ***************************************************** //
// ******************* Main Function ******************* //
// ***************************************************** //
int main(int argc, char **argv)
{
    //start logging into a file
    log_file = fopen("log.txt", "a");
    if (log_file == NULL){
        fprintf(stderr, "Error opening log file\n");
    }

    pthread_t t_ServerThread, t_MeasureThread, t_TimeThread;
    void *th_status;
    // kill the parent-process
    // and fork the current execution to force a new PID and
    // start as a daemon
    daemon(0, 0);

    // start the udp server task in background!

    // init all mutexes which are needed to block while dealing the run_t variable type (run)
    pthread_mutex_init( &run_meas_mutex, NULL );


    // create and start the threads
    if (pthread_create(&t_TimeThread, NULL, Time_thread, NULL) != 0) {
        #ifdef LOG_ACTIVITIES == 1
            log_activities("Problem occurred creating Time Thread", __LINE__);
        #endif
        return 1;
    }
    if( pthread_create( &t_ServerThread,        NULL,  Server_thread,   NULL ) !=0 ){
        #ifdef LOG_ACTIVITIES == 1
            log_activities("Problem occured creating Server Thread",__LINE__);
        #endif
        return 1;
    }
    if( pthread_create( &t_MeasureThread,       NULL,  Measure_thread,  NULL ) !=0 ){
        #ifdef LOG_ACTIVITIES == 1
            log_activities("Problem occured creating Measure Thread",__LINE__);
        #endif
        return 2;
    }


    // wait for two threads to terminate
    // go into blocked mode
    pthread_join( t_ServerThread,       &th_status );
    pthread_join( t_MeasureThread,      &th_status );


    // end main as last thread!
    run_thread = false; //this can cause problem in future
    pthread_join(t_TimeThread, &th_status);  // Wait for the time thread to finish
    pthread_mutex_destroy(&run_meas_mutex);
    fclose(log_file);
    pthread_exit(0);
    return( EXIT_SUCCESS );
}


// ***************************************************** //
// *********** Thread Function Declarations ************ //
// ***************************************************** //

/*
*       @brief          Thread works independantly from the program start to
*                               calculate the running time in mili seconds so that sync
*               can be done with the client
*       @name           ServerThread
*/
void *Time_thread() {
    struct timespec start, current;
    clock_gettime(CLOCK_MONOTONIC, &start);

    while (run_thread) {
        clock_gettime(CLOCK_MONOTONIC, &current);
        current_time = (current.tv_sec - start.tv_sec) * 1000.0 +
                       (current.tv_nsec - start.tv_nsec) / 1000000.0;
        usleep(1000); // Sleep for 1 millisecond
    }

    pthread_exit(NULL);
}


/*
*       @brief          Thread waits for UDP message(s) in blocked mode
*                               and parses the received message(s) and value(s).
*       @name           ServerThread
*/
void *Server_thread()
{
        char rx_buffer[40];

    //Initiate the udp server connection
    initiate_udp_connection();

        while( 1 ){
                memset( (void *)rx_buffer, (int)0, (size_t)(strlen( rx_buffer )) );

                // Go into blocked mode to free CPU; wait for UDP message
                if( recvfrom(udp.socket, rx_buffer, sizeof(rx_buffer), 0, (struct sockaddr *)&udp.client_addr, &udp.addr_length) < 0 ){
            #ifdef LOG_ACTIVITIES == 1
                log_activities("Recieving Message was not successfull.", __LINE__);
            #endif
                        break;
                }

        // Lock the mutex before changing the run
        pthread_mutex_lock(&run_meas_mutex);
                parse_udp_message( rx_buffer );
        pthread_mutex_unlock(&run_meas_mutex);
        }

        close( udp.socket );
        pthread_exit(NULL);
}


/*
*       @brief          Thread handles the data processing for signal from data acquisition to data holding
*                               and processes the udp.command(s).
*       @name           MeasureThread
*/
void *Measure_thread()
{
        // variables for measurement
    uint16_t *pData = ( uint16_t *)&(Header_with_data.p_data);


        if (initiate_redpitaya_with_dma() < 0){
        #ifdef LOG_ACTIVITIES == 1
            log_activities("Initiation of Redpitaya with DMA failed", __LINE__);
        #endif
        exit(1);
    }

    #ifdef LOG_ACTIVITIES == 1
        log_activities("Redpitaya Initiated", __LINE__);
    #endif

        // init all LEDs (LED0 - LED3)
        LEDx_INIT();

        // initiate the iic sensor
        initiate_iic();


        //Here the number of chunks are calculated and Cinstant for both cases data are being set to the Header
        Header_with_data.SampleFrequency        = (float)(ADC_SAMPLE_FREQUENCY); //sampling frequency of the ultrasonic signal [C]
    int Total_data_blocks_number        = (DATA_SIZE / READ_DATA_SIZE);
    if ((DATA_SIZE % READ_DATA_SIZE) != 0){
        Total_data_blocks_number += 1;
    }
        Header_with_data.TotalDataBlocks    = (float)(Total_data_blocks_number);
    Header_with_data.X_Interval = (float)(ADC_SAMPLE_TIME_NS); //This is the ADC sample time in nano seconds [C]
        Header_with_data.ADCResolution = (float)14; //Found from Redpitaya documentation sheet [C]
    Header_with_data.AmbTemperature = (float)15; //Should be a constant and measered by something else sensor [C]
    Header_with_data.Scaling = (float)1;
    Header_with_data.MeasDelay = (float)(ADC_START_DELAY_US * 1.5); //This is the ADC start Delay for only stopping capturing the sending pulse 8*64=512 ns / sample then 2 * 0,30 m / 343,2 m/s = 1.748 ms => 1748µs
    Header_with_data.SWVersion = (float)(SW_VERSION); //This is the constant software version


        // While loop - stay here while in "normal" operation
        while( 1 )
        {
                ( run != RUN_NONE )?( LED0_ON ):( LED0_OFF );

                // increment a counter; this is the time-base for all messurements and outputs [ms]
                // counter_ms = (counter_ms + THREAD_PAUSE_MS) % 1500;


                if (re_initiate_dma_channel() < 0){
            continue;
        }

        if (acquire_data() < 0){
            continue;
        }

        measure_distance(&distance);
        Header_with_data.CurrentTime = current_time; // Timestamp the data with acquiring finishing time
                // retval = measure_distance( &distance );
        #ifdef LOG_ACTIVITIES == 1
            log_activities("Data Acquired", __LINE__);
        #endif

        char log_message[150];
        sprintf(log_message, "Total Data Blocks in Header: %.2f", Header_with_data.TotalDataBlocks);
        #ifdef LOG_ACTIVITIES == 1
            log_activities(log_message, __LINE__);
        #endif

                // the "run" variable is (re-)set by the ServerThread via UDP
                switch(run)
                {
                        case RUN_NONE:
                //These are giving all the constant or variable type data of the header to be sent
                Header_with_data.HeaderLength = (float)(sizeof(Header_with_data) - sizeof(Header_with_data.p_data));
                Header_with_data.DataLength = (uint16_t)(sizeof(Header_with_data.p_data));

                //New data in header to be sent from previous stack
                Header_with_data.DataType =  (float) 0.0; //this has to be the data type currently only None so 0.0
                Header_with_data.Class = (float) 0.0; //this defines what type of object is infront of the sensor [V] currently not used
                send_binary( (uint8_t *)&Header_with_data, (uint16_t)(Header_with_data.HeaderLength) );
                                break;
                        case RUN_ADC:
                //The logic is select the chunk of data and send it, then select the next chunk
                Header_with_data.DataType =  (float) 1.0; //this has to be the data type currently only ADC so 1.0
                Header_with_data.Class = (float) 0.0;  //this defines what type of object is infront of the sensor [V] currently not used
                Header_with_data.FFTWindowLength = (float) 0.0; //These fields are conducting with FFT Data only FFT window length [V] currently not used and 0
                Header_with_data.FFTWindowOffsetIndex = (float) 0.0;  //These fields are conducting with FFT Data only FFT window Offset Index [V] currently not used and 0
                for (int block_number = 0; block_number < Total_data_blocks_number; block_number++){
                    Header_with_data.DataBlockNumber = (float) block_number;
                    read_data((int16_t *)pData, block_number);
                    Header_with_data.HeaderLength = (float)(sizeof(Header_with_data) - sizeof(Header_with_data.p_data));
                    Header_with_data.DataLength = (uint16_t)(sizeof(Header_with_data.p_data));

                    //New data in header to be sent from previous stack

                    Header_with_data.Distance = (float)(distance);

                    memcpy( (uint16_t *)Header_with_data.p_data, pData, READ_DATA_SIZE * sizeof(uint16_t));
                                send_binary( (uint8_t *)&Header_with_data, (uint16_t)(Header_with_data.HeaderLength + Header_with_data.DataLength) );
                    usleep(10);
                }
                            // send_binary( (uint8_t *)&Header_with_data, (uint16_t)(Header_with_data.HeaderLength) );
                pthread_mutex_lock(&run_meas_mutex);
                run = RUN_NONE;
                pthread_mutex_unlock(&run_meas_mutex);
                #ifdef LOG_ACTIVITIES == 1
                    log_activities("Data Sent Successfully", __LINE__);
                #endif
                                // send_data = true;
                                break;

                        default:
                                run = RUN_NONE;
                                break;
                }
        close_dma_channel();
        } // end while(1)

    release_resources();
        pthread_exit(NULL);
}

// ***************************************************** //
// *************** Function Declarations *************** //
// ***************************************************** //

/*
*@brief     This is the function to initiate UDP connection
*
*@details   This function initiates redpitaya's UDP
*           connection with necessary steps for socket
*           binding
*
*
*/
void initiate_udp_connection(){

    // First need to initiate Port and Address, family type
    udp.server_addr.sin_family = AF_INET;
    udp.server_addr.sin_port = htons( UDP_PORT );
    udp.server_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    // Save the client address length so that later on the process
    // we don't have mismatch with a new connection
    udp.addr_length = sizeof(udp.client_addr);

    //Create a socket with UDP protocol
    udp.socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

    // Check the socket is active or not
    if(udp.socket < 0){
        #ifdef LOG_ACTIVITIES == 1
            log_activities("Socket not active / properly connected", __LINE__);
        #endif
    }else{
        #ifdef LOG_ACTIVITIES == 1
            log_activities("Socket activated properly", __LINE__);
        #endif
    }

    if(bind(udp.socket, (struct sockaddr *)&udp.server_addr, sizeof(udp.server_addr)) < 0){
        #ifdef LOG_ACTIVITIES == 1
            log_activities( "Socket binding failed for UDP port ", __LINE__);
        #endif
    }else{
        #ifdef LOG_ACTIVITIES == 1
            log_activities("Socket binding successful for UDP port.", __LINE__);
        #endif
    }
}


/*
*@brief     This is the function to initiate redpitaya
*           with deep memory acquisition
*
*@details   This function initiates redpitaya, then
*           initiates necessary steps for deep memory
*           acquisition
*
*
*/
int initiate_redpitaya_with_dma(){
    /* Initialise Red Pitaya */
    if (rp_InitReset(false) != RP_OK) {
        #ifdef LOG_ACTIVITIES == 1
            log_activities("Rp api init failed!", __LINE__);
        #endif
        return -1;
    }

    /* Set Datapin0 as low or no use */
    rp_DpinSetDirection( RP_DIO0_P, RP_IN);     // DIO0 as input
    rp_DpinSetState( RP_DIO0_P, RP_LOW);        // DIO0 set to low


    /* Set up the Deep Memory Acquisition */
    uint32_t g_adc_axi_start, g_adc_axi_size;
    rp_AcqAxiGetMemoryRegion(&g_adc_axi_start, &g_adc_axi_size);
    char log_message[150];
    sprintf(log_message, "Reserved memory Start 0x%X Size 0x%X", g_adc_axi_start, g_adc_axi_size);
    #ifdef LOG_ACTIVITIES == 1
        log_activities(log_message, __LINE__);
    #endif

    /* Set decimation for both channels */
    if (rp_AcqAxiSetDecimationFactor(DECIMATION_FACTOR) != RP_OK) {
        #ifdef LOG_ACTIVITIES == 1
            log_activities("rp_AcqAxiSetDecimationFactor failed", __LINE__);
        #endif
        return -1;
    }

    /* Set trigger delay for channel */
    if (rp_AcqAxiSetTriggerDelay(RP_CH_1, DATA_SIZE)  != RP_OK) {
        #ifdef LOG_ACTIVITIES == 1
            log_activities("rp_AcqAxiSetTriggerDelay RP_CH_1 failed", __LINE__);
        #endif
        return -1;
    }

    /*
    Set-up the Channel 1 buffers to each work with half the available memory space.
    */
    if (rp_AcqAxiSetBufferSamples(RP_CH_1, g_adc_axi_start, DATA_SIZE) != RP_OK) {
        #ifdef LOG_ACTIVITIES == 1
            log_activities("rp_AcqAxiSetBuffer RP_CH_1 failed", __LINE__);
        #endif
        return -1;
    }

    /* Enable DMA on channel 1 */
    if (rp_AcqAxiEnable(RP_CH_1, true)) {
        #ifdef LOG_ACTIVITIES == 1
            log_activities("rp_AcqAxiEnable RP_CH_1 failed", __LINE__);
        #endif
        return -1;
    }
    #ifdef LOG_ACTIVITIES == 1
        log_activities("Enable CHA 1 at acquiring data", __LINE__);
    #endif
    /* Specify the acquisition trigger level*/
    rp_AcqSetTriggerLevel(RP_T_CH_1, 0);

    return 0;
}


/*
*@brief     This is the function to re initiate DMA channel
*           to avoid residual values in the ADC signal
*
*@details   This function re initiates DMA channel everytime before acquiring signal
*           otherwise residual value shifts the new signal after some time
*
*
*/
int re_initiate_dma_channel(){
    /* Enable DMA on channel 1 */
    if (rp_AcqAxiEnable(RP_CH_1, true)) {
        #ifdef LOG_ACTIVITIES == 1
            log_activities("rp_AcqAxiEnable RP_CH_1 failed", __LINE__);
        #endif
        return -1;
    }
    #ifdef LOG_ACTIVITIES == 1
        log_activities("Enable CHA 1", __LINE__);
    #endif

    return 0;
}

/*
*@brief     This is the function closes the DMA channel
*
*
*@details   This function closes DMA channel everytime before acquiring signal
*           otherwise residual value shifts the new signal after some time
*
*
*/
void close_dma_channel(){
    rp_AcqAxiEnable(RP_CH_1, false);
}


/*
*@brief     This is the function to initiate I2C
*           communication with sensor
*
*@details   This function initiates redpitaya's I square C
*           communication with the SONAR sensor
*
*
*/
void initiate_iic(){
    /* Open I²C port for reading and writing and also turning on the sensor */
    if ((iic.fd = open(iic.fileName, O_RDWR)) < 0) {
        #ifdef LOG_ACTIVITIES == 1
            log_activities("iic file opening failed", __LINE__);
        #endif
        exit(1);
    }

    /* Set the port options and set the address of the device we wish to speak to through I2C protocol */
    if (ioctl(iic.fd, I2C_SLAVE_FORCE, iic.address) < 0) {
        #ifdef LOG_ACTIVITIES == 1
            log_activities("iic address failed", __LINE__);
        #endif
        exit(1);
    }
}


/*
*@brief     This is the function to send sensor data via udp connection
*
*@details   This function only sends the data
*
*
*/
void send_binary( uint8_t*data_in, uint16_t count )
{
        //uint16_t Sent;
        if( udp.socket >= 0 ){
                sendto( udp.socket, (uint8_t *)data_in, count, MSG_DONTWAIT,  (struct sockaddr *) &udp.client_addr, udp.addr_length);
        #ifdef LOG_ACTIVITIES == 1
            log_activities("Message sent successfully", __LINE__);
        #endif
        }
}

/*
*@brief     This is the function to acquire sensor data
*
*@details   This function acquires the data from the sensor
*
*
*/
int acquire_data(){
    // I²C programming - start ultrasonic
    iic.buf[0] = 0;
    //iic.buf[1] = 0x51;        // measurement of distance
    iic.buf[1] = 0x52;          // measurement of time

    if ((write(iic.fd, iic.buf, 2)) != 2) {
        #ifdef LOG_ACTIVITIES == 1
            log_activities("iic buffering failed", __LINE__);
        #endif
        return -1;
    }

    // need to sleep for passing by the sending pulses
    usleep(ADC_START_DELAY_US * 1.5);

    /* Start the acquisition */
    if (rp_AcqStart() != RP_OK) {
        #ifdef LOG_ACTIVITIES == 1
            log_activities("rp_AcqStart failed", __LINE__);
        #endif
        return -1;
    }
    #ifdef LOG_ACTIVITIES == 1
        log_activities("ACQ Started", __LINE__);
    #endif

    /* Specify trigger source */
    rp_AcqSetTriggerSrc(RP_TRIG_SRC_CHA_PE);
    rp_acq_trig_state_t state = RP_TRIG_STATE_TRIGGERED;

    /* Wait for the triggering moment */
    while(1){
        rp_AcqGetTriggerState(&state);
        if(state == RP_TRIG_STATE_TRIGGERED){
            usleep(1000);
            break;
        }
    }

    /* Wait until buffer is full/data is acquired */
    bool fillState = false;
    while (!fillState) {
        if (rp_AcqAxiGetBufferFillState(RP_CH_1, &fillState) != RP_OK) {
            #ifdef LOG_ACTIVITIES == 1
                log_activities("rp_AcqAxiGetBufferFillState RP_CH_1 failed", __LINE__);
            #endif
            return -1;
        }
    }

    //Read data for getting time of flight of the signal
    do{
                usleep(500);
                read(iic.fd, iic.buf, 3 );
        }while( iic.buf[1]== 0xFF );

    time_of_flight_us = (float)( iic.buf[1]<<8 | iic.buf[2] ); // combining two bytes to form a single 16-bit value

    /* Stop the acquisition */
    rp_AcqStop();
    #ifdef LOG_ACTIVITIES == 1
        log_activities("Stop acq", __LINE__);
    #endif
    return 0;
}


/*
*@brief     This is the function to read acquired data
*
*
*@details   This function reads the data from the memory
*           It also selects which chunk should be sent
*
*/
void read_data(int16_t *data, int block_number){
    /* Get write pointer on the triggering location */
    uint32_t posChA;
    rp_AcqAxiGetWritePointerAtTrig(RP_CH_1, &posChA);

    uint32_t reading_point = (uint32_t) (block_number * READ_DATA_SIZE);
    posChA = reading_point;

    uint32_t size1 = READ_DATA_SIZE;

    rp_AcqAxiGetDataRaw(RP_CH_1, posChA, &size1, data);

}


/*
* @brief        This function measures the distance to an object
*
* @details      The function searches the maximum amplitude and
*                       its position.
*
*
*
*/
void measure_distance( float *distance_out )
{
        // calculate the distance of the measured object
        // and write back to distance_out pointer
        *distance_out = V_SONIC_WAVE * (time_of_flight_us / 2e6);
}

/*
*@brief     This is the function to release redpitaya memories
*
*@details   This function releases the redpitaya channel,
*           closes the udp socket, frees the memory and
*           release redpitaya initiation
*
*
*/
void release_resources(){

    /* Releasing resources */
    rp_AcqAxiEnable(RP_CH_1, false);
    //Close the udp_socket
    // close(udp.socket);

    rp_Release();
    //pthread_exit(NULL);

    //fclose(fp);
}


/*
*       @brief          This reads the message and sets the run variable
*                               as ADC data taking or Demo to send only the Header Data Length.
*       @name           parse_udp_message
*/
static void parse_udp_message( char *msg_rx )
{
        char *rx_param;

        if( msg_rx[0] == '-' ){

                udp.command = msg_rx[1];
                if( (rx_param = strstr( msg_rx, " ")) != NULL ){
                        udp.parameter[0] = atoi( rx_param++ );
                        if( (rx_param = strstr( rx_param, " ")) != NULL ){
                                udp.parameter[1] = atoi( rx_param++ );
                        }
                }

                switch( udp.command )
                {
                        case 'a':
                                run = RUN_ADC;
                                break;
            case 'i':
                run = RUN_NONE;
                break;
                        default:
                break;
                }
        }

}


/*
*@brief     This is the function to log activities
*
*@details   This function releases the redpitaya channel,
*           closes the udp socket, frees the memory and
*           release redpitaya initiation
*@name      log_activities
*
*/
void log_activities(char *message, int line){

    //Get the current time
    time_t t;
    time(&t);

    if (fprintf(log_file, "%s:(Line: %d) %s.\n", ctime(&t), line, message) < 0){
        fprintf(stderr, "Error writing to log file\n");
    };
}