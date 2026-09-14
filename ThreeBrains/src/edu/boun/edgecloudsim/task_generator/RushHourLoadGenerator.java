/*
 * Title:        EdgeCloudSim - Rush Hour Load Generator
 * Description:  Implements a time-varying task generation rate (NHPP).
 * Includes SimLogger for validation and debugging.
 */

package edu.boun.edgecloudsim.task_generator;

import java.util.ArrayList;
import org.apache.commons.math3.distribution.ExponentialDistribution;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.SimUtils;
import edu.boun.edgecloudsim.utils.TaskProperty;

public class RushHourLoadGenerator extends LoadGeneratorModel {
    int taskTypeOfDevices[];

    // Rush Hour Configuration
    // You can also read these from SimSettings if you add them to config.properties
    private double rushHourStart = 300;       // Start at 20 minutes
    private double rushHourDuration = 600;    // Lasts for 20 minutes
    private double rushHourLoadMultiplier = 3.0; // 3x Load Intensity

    public RushHourLoadGenerator(int _numberOfMobileDevices, double _simulationTime, String _simScenario) {
        super(_numberOfMobileDevices, _simulationTime, _simScenario);
    }

    @Override
    public void initializeModel() {
        taskList = new ArrayList<TaskProperty>();
        
        // LOGGING: Confirm initialization parameters
        SimLogger.printLine("----------------------------------------------------------------------");
        SimLogger.printLine("RushHourLoadGenerator Initialized");
        SimLogger.printLine("Rush Hour Start: " + rushHourStart + "s");
        SimLogger.printLine("Rush Hour End: " + (rushHourStart + rushHourDuration) + "s");
        SimLogger.printLine("Load Multiplier: " + rushHourLoadMultiplier + "x");
        SimLogger.printLine("----------------------------------------------------------------------");

        double rushHourIntervalFactor = 1.0 / rushHourLoadMultiplier;

        // 1. Initialize RNGs for Task Characteristics
        ExponentialDistribution[][] expRngList = new ExponentialDistribution[SimSettings.getInstance().getTaskLookUpTable().length][3];
        
        for(int i = 0; i < SimSettings.getInstance().getTaskLookUpTable().length; i++) {
            if(SimSettings.getInstance().getTaskLookUpTable()[i][0] == 0) continue;
            
            expRngList[i][0] = new ExponentialDistribution(SimSettings.getInstance().getTaskLookUpTable()[i][5]);
            expRngList[i][1] = new ExponentialDistribution(SimSettings.getInstance().getTaskLookUpTable()[i][6]);
            expRngList[i][2] = new ExponentialDistribution(SimSettings.getInstance().getTaskLookUpTable()[i][7]);
        }
        
        // 2. Assign Task Types to Devices
        taskTypeOfDevices = new int[numberOfMobileDevices];
        for(int i = 0; i < numberOfMobileDevices; i++) {
            int randomTaskType = -1;
            double taskTypeSelector = SimUtils.getRandomDoubleNumber(0, 100);
            double taskTypePercentage = 0;
            
            for (int j = 0; j < SimSettings.getInstance().getTaskLookUpTable().length; j++) {
                taskTypePercentage += SimSettings.getInstance().getTaskLookUpTable()[j][0];
                if(taskTypeSelector <= taskTypePercentage){
                    randomTaskType = j;
                    break;
                }
            }
            
            // LOGGING: Critical Error Check
            if(randomTaskType == -1){
                SimLogger.printLine("Critical Error: No valid task type assigned to device " + i + "!");
                continue;
            }

            taskTypeOfDevices[i] = randomTaskType;
            
            double poissonMean = SimSettings.getInstance().getTaskLookUpTable()[randomTaskType][2];
            double activePeriod = SimSettings.getInstance().getTaskLookUpTable()[randomTaskType][3];
            double idlePeriod = SimSettings.getInstance().getTaskLookUpTable()[randomTaskType][4];
            
            double activePeriodStartTime = SimUtils.getRandomDoubleNumber(
                    SimSettings.CLIENT_ACTIVITY_START_TIME, 
                    SimSettings.CLIENT_ACTIVITY_START_TIME + activePeriod);
            double virtualTime = activePeriodStartTime;

            ExponentialDistribution rng = new ExponentialDistribution(poissonMean);
            
            // Counter for verification (optional)
            int normalTasks = 0;
            int rushHourTasks = 0;

            while(virtualTime < simulationTime) {
                double interval = rng.sample();
                
                // LOGGING: Warning for bad RNG
                if(interval <= 0){
                    SimLogger.printLine("Warning: Invalid interval " + interval + " for device " + i + " at time " + virtualTime);
                    continue;
                }

                // --- Rush Hour Logic ---
                boolean isRushHour = (virtualTime >= rushHourStart) && 
                                     (virtualTime < (rushHourStart + rushHourDuration));
                
                if (isRushHour) {
                    interval = interval * rushHourIntervalFactor;
                    rushHourTasks++;
                } else {
                    normalTasks++;
                }
                
                virtualTime += interval;
                
                if(virtualTime > activePeriodStartTime + activePeriod){
                    activePeriodStartTime = activePeriodStartTime + activePeriod + idlePeriod;
                    virtualTime = activePeriodStartTime;
                    continue;
                }
                
                // Add the task
                taskList.add(new TaskProperty(i, randomTaskType, virtualTime, expRngList));
            }
            
            // VERIFICATION LOG: Print stats for the first device only to keep logs clean
            if (i == 0) {
                 SimLogger.printLine("Device 0 Stats: Normal Tasks=" + normalTasks + ", RushHour Tasks=" + rushHourTasks);
                 if (rushHourTasks == 0) {
                     SimLogger.printLine("WARNING: No Rush Hour tasks generated. Check simulation_time vs rush_hour_start.");
                 }
            }
        }
    }

    @Override
    public int getTaskTypeOfDevice(int deviceId) {
        return taskTypeOfDevices[deviceId];
    }
}