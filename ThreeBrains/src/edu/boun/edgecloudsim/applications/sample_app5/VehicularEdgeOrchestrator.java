/*
 * Title:        EdgeCloudSim - Edge Orchestrator implementation
 * 
 * Description: 
 * VehicularEdgeOrchestrator decides which tier (mobile, edge or cloud)
 * to offload and picks proper VM to execute incoming tasks
 *               
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 * Copyright (c) 2017, Bogazici University, Istanbul, Turkey
 */
package edu.boun.edgecloudsim.applications.sample_app5;

import java.util.stream.DoubleStream;

import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEvent;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.core.SimSettings.NETWORK_DELAY_TYPES;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_server.EdgeHost;
import edu.boun.edgecloudsim.edge_server.EdgeVM;
import edu.boun.edgecloudsim.utils.Location;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.SimUtils;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import org.cloudbus.cloudsim.Datacenter;

public class VehicularEdgeOrchestrator extends EdgeOrchestrator {

    private static final int BASE = 100000; //start from base in order not to conflict cloudsim tag
    private static final int UPDATE_PREDICTION_WINDOW = BASE + 1;
    private static final int UPDATE_OPTIMIZER = BASE + 2; // Tag for optimization event

    public static final int CLOUD_DATACENTER_VIA_GSM = 1;
    public static final int CLOUD_DATACENTER_VIA_RSU = 2;
    public static final int EDGE_DATACENTER = 3;

   
    private static final double DELTA_T = 60.0;

    private static final String OPTIMIZER_URL = "http://127.0.0.1:5001/optimize";
    private static final String RT_OPT_ML_URL = "http://127.0.0.1:5002/optimize";

    private static final String EXP3_MC_POLICY = "EXP3_MC";
    private static final String EXP3_MC_URL = "http://127.0.0.1:5003/optimize";
   
    private static final String SCOPE_POLICY = "SCOPE";
    private static final String SOFTCOST_OPT_POLICY = "SOFTCOST_OPT";
    private static final String PERFORMANCE_ONLY_POLICY = "PERFORMANCE_ONLY";


    private static final String SCOPE_DATA_COLLECTION_POLICY = "SCOPE_DATA_COLLECTION";
    private static final String TRAINING_DATA_PATH = System.getProperty(
            "scope.trainingData.path", "scope_candidate_windows_raw.csv");
    private static final long CANDIDATE_BASE_SEED = Long.parseLong(
            System.getProperty("scope.trainingData.seed", "20260827"));

    private int cloudVmCounter;
    private int edgeVmCounter;
    private int numOfMobileDevice;

    private OrchestratorStatisticLogger statisticLogger;
    private OrchestratorTrainerLogger trainerLogger;

    private MultiArmedBanditHelper MAB;
    private GameTheoryHelper GTH;

    private static final int NUM_APP_TYPES = 3;
    private double[] wLoadPerType = new double[NUM_APP_TYPES];
    private double[] wDistPerType = new double[NUM_APP_TYPES];
    private double[] thresholdPerType = new double[NUM_APP_TYPES];


    private double currentAlpha = 0.6;

 
    private Map<String, Integer> taskDecisionMap = new ConcurrentHashMap<>();
    private int[] lagTasksPerType = new int[NUM_APP_TYPES]; //  Lag Tracking

    // Counters for the current window
    private int sentEdge = 0, sentRSU = 0, sentGSM = 0;
    private int failEdge = 0, failRSU = 0, failGSM = 0;

    // For Churn Calc
    private double totalChurnAccumulated = 0;
    private double[] prevWLoad = new double[NUM_APP_TYPES];
    private double[] prevThreshold = new double[NUM_APP_TYPES];
    private double prevAlpha = 0.6;

    // Statistics Window
    private int[] failuresPerType = new int[NUM_APP_TYPES];
    private int[] tasksPerType = new int[NUM_APP_TYPES];
    private int[] completedPerType = new int[NUM_APP_TYPES];
    private int[] deadlineMissesPerType = new int[NUM_APP_TYPES];
    private double[] serviceTimeSumPerType = new double[NUM_APP_TYPES];

    private final Map<String, Long> taskTrainingWindowMap = new ConcurrentHashMap<>();
    private final Map<Long, CandidateTrainingWindow> trainingWindows = new LinkedHashMap<>();
    private long nextTrainingWindowId = 0;
    private long activeTrainingWindowId = -1;
    private Random candidateRandom;
    private String trainingRunId;

    private Random simulationRandom;

    private static class OptimizerStateSnapshot {

        double timestamp;
        int[] total = new int[NUM_APP_TYPES];
        int[] lagTotal = new int[NUM_APP_TYPES];
        int[] completed = new int[NUM_APP_TYPES];
        int[] failed = new int[NUM_APP_TYPES];
        int[] deadlineMisses = new int[NUM_APP_TYPES];
        double[] failureRate = new double[NUM_APP_TYPES];
        double[] meanServiceTime = new double[NUM_APP_TYPES];
        double[] deadlineNormServiceTime = new double[NUM_APP_TYPES];
        double[] deadlineMissRate = new double[NUM_APP_TYPES];
        double avgEdgeCpu;
        double maxEdgeCpu;
        int sentEdge;
        int sentRsu;
        int sentGsm;
        int failEdge;
        int failRsu;
        int failGsm;
        double edgeFailureRate;
        double rsuFailureRate;
        double gsmFailureRate;
        double previousAlpha;
        double[] previousWLoad = new double[NUM_APP_TYPES];
        double[] previousThreshold = new double[NUM_APP_TYPES];
    }

    private static class CandidateConfiguration {

        double alpha;
        double[] wLoad = new double[NUM_APP_TYPES];
        double[] threshold = new double[NUM_APP_TYPES];
    }

    private static class CandidateTrainingWindow {

        long id;
        double startTime;
        double endTime;
        boolean closed;
        OptimizerStateSnapshot inputState;
        CandidateConfiguration candidate;
        int[] generated = new int[NUM_APP_TYPES];
        int[] completed = new int[NUM_APP_TYPES];
        int[] failed = new int[NUM_APP_TYPES];
        int[] deadlineMisses = new int[NUM_APP_TYPES];
        double[] serviceTimeSum = new double[NUM_APP_TYPES];
    }
    // ===== SCOPE DATA COLLECTION CHANGE END =====

    private static final double FAST_TRACK_SPIKE_THRESHOLD = 20.0;
    private double utilizationAtWindowStart = 0.0;
    private boolean fastTrackFiredThisWindow = false;

    /**
     * Constructor for vehicular edge orchestrator
     *
     * @param _numOfMobileDevices Number of mobile devices in simulation
     * @param _policy Orchestration policy (AI_BASED, GAME_THEORY, MAB, etc.)
     * @param _simScenario Simulation scenario type
     */
    public VehicularEdgeOrchestrator(int _numOfMobileDevices, String _policy, String _simScenario) {
        super(_policy, _simScenario);
        this.numOfMobileDevice = _numOfMobileDevices;
    }

    @Override
    public void initialize() {
        // Initialize VM counters for load balancing
        cloudVmCounter = 0;
        edgeVmCounter = 0;
        Arrays.fill(lagTasksPerType, 0);
        Arrays.fill(tasksPerType, 0);
        Arrays.fill(completedPerType, 0);
        Arrays.fill(failuresPerType, 0);
        Arrays.fill(deadlineMissesPerType, 0);
        Arrays.fill(serviceTimeSumPerType, 0.0);
        totalChurnAccumulated = 0;

        trainingRunId = System.getProperty("scope.runId", "run_unspecified");
        long derivedSeed = CANDIDATE_BASE_SEED
                + (31L * numOfMobileDevice)
                + policy.hashCode()
                + trainingRunId.hashCode();
        candidateRandom = new Random(derivedSeed);

        // Initialize logging components
        statisticLogger = new OrchestratorStatisticLogger();
        trainerLogger = new OrchestratorTrainerLogger();

        // Initialize Multi-Armed Bandit helper with task length bounds
        double lookupTable[][] = SimSettings.getInstance().getTaskLookUpTable();
        // Assume first app has lowest and last app has highest task length
        double minTaskLength = lookupTable[0][7];
        double maxTaskLength = lookupTable[lookupTable.length - 1][7];
        MAB = new MultiArmedBanditHelper(minTaskLength, maxTaskLength);

        // Initialize Game Theory helper for Nash equilibrium calculations
        // Parameters: min load (0), max load (20), number of players (mobile devices)
        GTH = new GameTheoryHelper(0, 20, numOfMobileDevice);

        // Initialize Defaults
        wLoadPerType[0] = 0.5;
        wDistPerType[0] = 0.5;
        thresholdPerType[0] = 85.0;
        wLoadPerType[1] = 0.7;
        wDistPerType[1] = 0.3;
        thresholdPerType[1] = 80.0;
        wLoadPerType[2] = 0.4;
        wDistPerType[2] = 0.6;
        thresholdPerType[2] = 90.0;

        this.currentAlpha = 0.6;

        long simulationSeed = Long.parseLong(
                System.getProperty("scope.simSeed", "20260828")
        );

        
        simulationRandom = new Random(simulationSeed + 97531L);

        System.arraycopy(wLoadPerType, 0, prevWLoad, 0, NUM_APP_TYPES);
        System.arraycopy(thresholdPerType, 0, prevThreshold, 0, NUM_APP_TYPES);
    }

    /**
     * Determines which computing tier to offload task based on orchestration
     * policy
     *
     * @param task Task to be offloaded
     * @return Target device ID (EDGE_DATACENTER, CLOUD_DATACENTER_VIA_RSU, or
     * CLOUD_DATACENTER_VIA_GSM)
     */
    @Override
    public int getDeviceToOffload(Task task) {
        int result = 0;

        int type = task.getTaskType();
        if (usesRtOptLogic() && type >= 0 && type < NUM_APP_TYPES) {
            tasksPerType[type]++;
        }

        // Get current resource utilization metrics
        double avgEdgeUtilization = SimManager.getInstance().getEdgeServerManager().getAvgUtilization();
        double avgCloudUtilization = SimManager.getInstance().getCloudServerManager().getAvgUtilization();

        // Estimate network delays for different communication paths
        VehicularNetworkModel networkModel = (VehicularNetworkModel) SimManager.getInstance().getNetworkModel();
        double wanUploadDelay = networkModel.estimateUploadDelay(NETWORK_DELAY_TYPES.WAN_DELAY, task);
        double wanDownloadDelay = networkModel.estimateDownloadDelay(NETWORK_DELAY_TYPES.WAN_DELAY, task);

        double gsmUploadDelay = networkModel.estimateUploadDelay(NETWORK_DELAY_TYPES.GSM_DELAY, task);
        double gsmDownloadDelay = networkModel.estimateDownloadDelay(NETWORK_DELAY_TYPES.GSM_DELAY, task);

        double wlanUploadDelay = networkModel.estimateUploadDelay(NETWORK_DELAY_TYPES.WLAN_DELAY, task);
        double wlanDownloadDelay = networkModel.estimateDownloadDelay(NETWORK_DELAY_TYPES.WLAN_DELAY, task);

        // Available offloading options
        int options[] = {
            EDGE_DATACENTER,
            CLOUD_DATACENTER_VIA_RSU,
            CLOUD_DATACENTER_VIA_GSM
        };

        // Handle zero delays for AI-based and advanced algorithms
        // Replace zero delays with maximum values to indicate unavailability
        if (policy.startsWith("AI_") || policy.equals("MAB") || policy.equals("GAME_THEORY")) {
            if (wanUploadDelay == 0) {
                wanUploadDelay = WekaWrapper.MAX_WAN_DELAY;
            }

            if (wanDownloadDelay == 0) {
                wanDownloadDelay = WekaWrapper.MAX_WAN_DELAY;
            }

            if (gsmUploadDelay == 0) {
                gsmUploadDelay = WekaWrapper.MAX_GSM_DELAY;
            }

            if (gsmDownloadDelay == 0) {
                gsmDownloadDelay = WekaWrapper.MAX_GSM_DELAY;
            }

            if (wlanUploadDelay == 0) {
                wlanUploadDelay = WekaWrapper.MAX_WLAN_DELAY;
            }

            if (wlanDownloadDelay == 0) {
                wlanDownloadDelay = WekaWrapper.MAX_WLAN_DELAY;
            }
        }

        if (policy.equals("RAIDER_TRAX") || usesRtOptLogic()) {
            // ===== SCOPE DATA COLLECTION CHANGE END =====
            // 1. Try Edge
            EdgeVM bestEdgeVm = findBestEdgeServer(task, true);

            if (bestEdgeVm != null) {
                result = EDGE_DATACENTER;
            } else {
                // 2. Fallback
                double probRSU = 0.6; // RAIDER_TRAX Default
                if (usesRtOptLogic()) {
                    probRSU = currentAlpha;
                }
                if (simulationRandom.nextDouble() < probRSU) {
                    result = CLOUD_DATACENTER_VIA_RSU;
                } else {
                    result = CLOUD_DATACENTER_VIA_GSM;
                }
            }

            // Track Decisions
            if (!policy.equals("RAIDER_TRAX")) {
                String key = task.getMobileDeviceId() + "_" + task.getCloudletId();
                taskDecisionMap.put(key, result);
                if (result == EDGE_DATACENTER) {
                    sentEdge++;
                } else if (result == CLOUD_DATACENTER_VIA_RSU) {
                    sentRSU++;
                } else {
                    sentGSM++;
                }

                if (isTrainingDataActive() && activeTrainingWindowId >= 0
                        && type >= 0 && type < NUM_APP_TYPES) {
                    CandidateTrainingWindow trainingWindow = trainingWindows.get(activeTrainingWindowId);
                    if (trainingWindow != null) {
                        trainingWindow.generated[type]++;
                        taskTrainingWindowMap.put(key, activeTrainingWindowId);
                    }
                }
            }
        } 

        else if (policy.equals("AI_BASED")) {
            WekaWrapper weka = WekaWrapper.getInstance();

            // Classify success probability for edge datacenter
            boolean predictedResultForEdge = weka.handleClassification(EDGE_DATACENTER,
                    new double[]{trainerLogger.getOffloadStat(EDGE_DATACENTER - 1),
                        task.getCloudletLength(), wlanUploadDelay,
                        wlanDownloadDelay, avgEdgeUtilization});

            // Classify success probability for cloud via RSU
            boolean predictedResultForCloudViaRSU = weka.handleClassification(CLOUD_DATACENTER_VIA_RSU,
                    new double[]{trainerLogger.getOffloadStat(CLOUD_DATACENTER_VIA_RSU - 1),
                        wanUploadDelay, wanDownloadDelay});

            // Classify success probability for cloud via GSM
            boolean predictedResultForCloudViaGSM = weka.handleClassification(CLOUD_DATACENTER_VIA_GSM,
                    new double[]{trainerLogger.getOffloadStat(CLOUD_DATACENTER_VIA_GSM - 1),
                        gsmUploadDelay, gsmDownloadDelay});

            // Initialize service time predictions with maximum values (infeasible)
            double predictedServiceTimeForEdge = Double.MAX_VALUE;
            double predictedServiceTimeForCloudViaRSU = Double.MAX_VALUE;
            double predictedServiceTimeForCloudViaGSM = Double.MAX_VALUE;

            // Predict service times only for options classified as successful
            if (predictedResultForEdge) {
                predictedServiceTimeForEdge = weka.handleRegression(EDGE_DATACENTER,
                        new double[]{task.getCloudletLength(), avgEdgeUtilization});
            }

            if (predictedResultForCloudViaRSU) {
                predictedServiceTimeForCloudViaRSU = weka.handleRegression(CLOUD_DATACENTER_VIA_RSU,
                        new double[]{task.getCloudletLength(), wanUploadDelay, wanDownloadDelay});
            }

            if (predictedResultForCloudViaGSM) {
                predictedServiceTimeForCloudViaGSM = weka.handleRegression(CLOUD_DATACENTER_VIA_GSM,
                        new double[]{task.getCloudletLength(), gsmUploadDelay, gsmDownloadDelay});
            }

            // Handle case where no option is predicted as successful - random selection
            if (!predictedResultForEdge && !predictedResultForCloudViaRSU && !predictedResultForCloudViaGSM) {
                double probabilities[] = {0.33, 0.34, 0.33};

                double randomNumber = edu.boun.edgecloudsim.utils.SimUtils.getRandomDoubleNumber(0, 1);
                double lastPercentagte = 0;
                boolean resultFound = false;
                for (int i = 0; i < probabilities.length; i++) {
                    if (randomNumber <= probabilities[i] + lastPercentagte) {
                        result = options[i];
                        resultFound = true;
                        break;
                    }
                    lastPercentagte += probabilities[i];
                }

                if (!resultFound) {
                    SimLogger.printLine("Unexpected probability calculation! Terminating simulation...");
                    System.exit(1);
                }
            } // Select option with minimum predicted service time
            else if (predictedServiceTimeForEdge <= Math.min(predictedServiceTimeForCloudViaRSU, predictedServiceTimeForCloudViaGSM)) {
                result = EDGE_DATACENTER;
            } else if (predictedServiceTimeForCloudViaRSU <= Math.min(predictedServiceTimeForEdge, predictedServiceTimeForCloudViaGSM)) {
                result = CLOUD_DATACENTER_VIA_RSU;
            } else if (predictedServiceTimeForCloudViaGSM <= Math.min(predictedServiceTimeForEdge, predictedServiceTimeForCloudViaRSU)) {
                result = CLOUD_DATACENTER_VIA_GSM;
            } else {
                SimLogger.printLine("Impossible occurred in AI based algorithm! Terminating simulation...");
                System.exit(1);
            }

            // Update offload statistics for future predictions
            trainerLogger.addOffloadStat(result - 1);
        } // AI_TRAINER: Generate training data with task-type-specific probability distributions
        else if (policy.equals("AI_TRAINER")) {
            double probabilities[] = null;
            // Different probabilities based on task type for diverse training data
            if (task.getTaskType() == 0) {
                probabilities = new double[]{0.60, 0.23, 0.17}; // Edge-heavy for task type 0
            } else if (task.getTaskType() == 1) {
                probabilities = new double[]{0.30, 0.53, 0.17}; // Cloud-via-RSU heavy for task type 1
            } else {
                probabilities = new double[]{0.23, 0.60, 0.17}; // Cloud-via-GSM heavy for other types
            }
            double randomNumber = edu.boun.edgecloudsim.utils.SimUtils.getRandomDoubleNumber(0, 1);
            double lastPercentagte = 0;
            boolean resultFound = false;
            for (int i = 0; i < probabilities.length; i++) {
                if (randomNumber <= probabilities[i] + lastPercentagte) {
                    result = options[i];
                    resultFound = true;

                    // Record training data with network delays and decision
                    trainerLogger.addStat(task.getCloudletId(), result,
                            wanUploadDelay, wanDownloadDelay,
                            gsmUploadDelay, gsmDownloadDelay,
                            wlanUploadDelay, wlanDownloadDelay);

                    break;
                }
                lastPercentagte += probabilities[i];
            }

            if (!resultFound) {
                SimLogger.printLine("Unexpected probability calculation for AI based orchestrator! Terminating simulation...");
                System.exit(1);
            }
        } // RANDOM: Uniform random selection among all options (baseline)
        else if (policy.equals("RANDOM")) {
            double probabilities[] = {0.33, 0.33, 0.34};

            // double randomNumber = edu.boun.edgecloudsim.utils.SimUtils.getRandomDoubleNumber(0, 1);
            double randomNumber = simulationRandom.nextDouble();

            double lastPercentagte = 0;
            boolean resultFound = false;
            for (int i = 0; i < probabilities.length; i++) {
                if (randomNumber <= probabilities[i] + lastPercentagte) {
                    result = options[i];
                    resultFound = true;
                    break;
                }
                lastPercentagte += probabilities[i];
            }

            if (!resultFound) {
                SimLogger.printLine("Unexpected probability calculation for random orchestrator! Terminating simulation...");
                System.exit(1);
            }
        } // MAB: Multi-Armed Bandit with Upper Confidence Bound algorithm
        else if (policy.equals("MAB")) {
            // Initialize MAB with expected delays if not already done
            if (!MAB.isInitialized()) {
                double expectedProcessingDealyOnCloud = task.getCloudletLength()
                        / SimSettings.getInstance().getMipsForCloudVM();

                // All Edge VMs are identical, get MIPS from first VM
                double expectedProcessingDealyOnEdge = task.getCloudletLength()
                        / SimManager.getInstance().getEdgeServerManager().getVmList(0).get(0).getMips();

                // Calculate total expected delays for each option
                double[] expectedDelays = {
                    wlanUploadDelay + wlanDownloadDelay + expectedProcessingDealyOnEdge,
                    wanUploadDelay + wanDownloadDelay + expectedProcessingDealyOnCloud,
                    gsmUploadDelay + gsmDownloadDelay + expectedProcessingDealyOnCloud
                };

                MAB.initialize(expectedDelays, task.getCloudletLength());
            }

            // Use UCB algorithm to select arm (option) with highest upper confidence bound
            result = options[MAB.runUCB(task.getCloudletLength())];
        } // GAME_THEORY: Nash equilibrium-based decision using game theory
        else if (policy.equals("GAME_THEORY")) {
            // Calculate expected processing delay on edge considering current utilization
            double expectedProcessingDealyOnEdge = task.getCloudletLength()
                    / SimManager.getInstance().getEdgeServerManager().getVmList(0).get(0).getMips();

            // Adjust for queueing delay based on utilization (M/M/1 queue approximation)
            expectedProcessingDealyOnEdge *= 100 / (100 - avgEdgeUtilization);

            double expectedEdgeDelay = expectedProcessingDealyOnEdge
                    + wlanUploadDelay + wlanDownloadDelay;

            // Calculate expected processing delay on cloud considering utilization
            double expectedProcessingDealyOnCloud = task.getCloudletLength()
                    / SimSettings.getInstance().getMipsForCloudVM();

            expectedProcessingDealyOnCloud *= 100 / (100 - avgCloudUtilization);

            // Randomly choose between GSM and RSU for cloud access
            boolean isGsmFaster = edu.boun.edgecloudsim.utils.SimUtils.getRandomDoubleNumber(0, 1) < 0.5;
            double expectedCloudDelay = expectedProcessingDealyOnCloud
                    + (isGsmFaster ? gsmUploadDelay : wanUploadDelay)
                    + (isGsmFaster ? gsmDownloadDelay : wanDownloadDelay);

            // Get task-specific parameters for game theory calculation
            double taskArrivalRate = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][2];
            double maxDelay = SimSettings.getInstance().getTaskLookUpTable()[task.getTaskType()][13] * (double) 6;

            // Calculate Nash equilibrium probability for this mobile device
            double Pi = GTH.getPi(task.getMobileDeviceId(), taskArrivalRate, expectedEdgeDelay, expectedCloudDelay, maxDelay);

            double randomNumber = edu.boun.edgecloudsim.utils.SimUtils.getRandomDoubleNumber(0, 1);

            // Make decision based on Nash equilibrium probability
            if (Pi < randomNumber) {
                result = EDGE_DATACENTER;
            } else {
                result = (isGsmFaster ? CLOUD_DATACENTER_VIA_GSM : CLOUD_DATACENTER_VIA_RSU);
            }
        } // PREDICTIVE: Adaptive algorithm based on historical performance statistics
        else if (policy.equals("PREDICTIVE")) {
            // Initial uniform probability distribution
            double probabilities[] = {0.34, 0.33, 0.33};

            // Use predictive logic only after warm-up period to collect statistics
            if (CloudSim.clock() > SimSettings.getInstance().getWarmUpPeriod()) {
                // Calculate failure rates for each option
                double failureRates[] = {
                    statisticLogger.getFailureRate(options[0]),
                    statisticLogger.getFailureRate(options[1]),
                    statisticLogger.getFailureRate(options[2])
                };

                // Calculate average service times for each option
                double serviceTimes[] = {
                    statisticLogger.getServiceTime(options[0]),
                    statisticLogger.getServiceTime(options[1]),
                    statisticLogger.getServiceTime(options[2])
                };

                double failureRateScores[] = {0, 0, 0};
                double serviceTimeScores[] = {0, 0, 0};

                // Calculate inverse scores (lower values get higher scores)
                for (int i = 0; i < probabilities.length; i++) {
                    // Inverse failure rate scoring: lower failure rate = higher score
                    failureRateScores[i] = DoubleStream.of(failureRates).sum() / failureRates[i];
                    // Inverse service time scoring: lower service time = higher score
                    serviceTimeScores[i] = DoubleStream.of(serviceTimes).sum() / serviceTimes[i];
                }

                // Adapt probabilities based on performance metrics
                for (int i = 0; i < probabilities.length; i++) {
                    // Prioritize failure rate if overall failure rate is high (>30%)
                    if (DoubleStream.of(failureRates).sum() > 0.3) {
                        probabilities[i] = failureRateScores[i] / DoubleStream.of(failureRateScores).sum();
                    } else // Otherwise prioritize service time optimization
                    {
                        probabilities[i] = serviceTimeScores[i] / DoubleStream.of(serviceTimeScores).sum();
                    }
                }
            }

            double randomNumber = edu.boun.edgecloudsim.utils.SimUtils.getRandomDoubleNumber(0.01, 0.99);
            double lastPercentagte = 0;
            boolean resultFound = false;
            for (int i = 0; i < probabilities.length; i++) {
                if (randomNumber <= probabilities[i] + lastPercentagte) {
                    result = options[i];
                    resultFound = true;
                    break;
                }
                lastPercentagte += probabilities[i];
            }

            if (!resultFound) {
                SimLogger.printLine("Unexpected probability calculation for predictive orchestrator! Terminating simulation...");
                System.exit(1);
            }
        } else {
            SimLogger.printLine("Unknow edge orchestrator policy! Terminating simulation...");
            System.exit(1);
        }

        return result;
    }

    /**
     * Selects a specific VM within the chosen datacenter using round-robin load
     * balancing
     *
     * @param task Task to be assigned to VM
     * @param deviceId Target datacenter ID (edge or cloud)
     * @return Selected VM instance for task execution
     */
    @Override
    public Vm getVmToOffload(Task task, int deviceId) {
        Vm selectedVM = null;

        
        if ((policy.equals("RAIDER_TRAX") || usesRtOptLogic()) && deviceId == EDGE_DATACENTER) {
            selectedVM = findBestEdgeServer(task, false);
            if (selectedVM == null) {
                try {
                    selectedVM = (EdgeVM) SimManager.getInstance().getEdgeServerManager().getDatacenterList().get(0).getHostList().get(0).getVmList().get(0);
                } catch (Exception e) {
                }
            }
            return selectedVM;
        }
        
        // Cloud VM selection (both GSM and RSU use same cloud infrastructure)
        if (deviceId == CLOUD_DATACENTER_VIA_GSM || deviceId == CLOUD_DATACENTER_VIA_RSU) {
            int numOfCloudHosts = SimSettings.getInstance().getNumOfCloudHost();
            int hostIndex = (cloudVmCounter / numOfCloudHosts) % numOfCloudHosts;
            int vmIndex = cloudVmCounter % SimSettings.getInstance().getNumOfCloudVMsPerHost();;

            selectedVM = SimManager.getInstance().getCloudServerManager().getVmList(hostIndex).get(vmIndex);

            // Round-robin cloud VM counter
            cloudVmCounter++;
            cloudVmCounter = cloudVmCounter % SimSettings.getInstance().getNumOfCloudVMs();

        } // Edge VM selection
        else if (deviceId == EDGE_DATACENTER) {
            int numOfEdgeVMs = SimSettings.getInstance().getNumOfEdgeVMs();
            int numOfEdgeHosts = SimSettings.getInstance().getNumOfEdgeHosts();
            int vmPerHost = numOfEdgeVMs / numOfEdgeHosts;

            int hostIndex = (edgeVmCounter / vmPerHost) % numOfEdgeHosts;
            int vmIndex = edgeVmCounter % vmPerHost;

            selectedVM = SimManager.getInstance().getEdgeServerManager().getVmList(hostIndex).get(vmIndex);

            // Round-robin edge VM counter
            edgeVmCounter++;
            edgeVmCounter = edgeVmCounter % numOfEdgeVMs;
        } else {
            SimLogger.printLine("Unknown device id! Terminating simulation...");
            System.exit(1);
        }
        return selectedVM;
    }

    @Override
    public void startEntity() {
        // Schedule periodic statistics window updates for predictive policy
        if (policy.equals("PREDICTIVE")) {
            schedule(getId(), SimSettings.CLIENT_ACTIVITY_START_TIME
                    + OrchestratorStatisticLogger.PREDICTION_WINDOW_UPDATE_INTERVAL,
                    UPDATE_PREDICTION_WINDOW);
        }

        // Start Optimizer Loop for Dynamic Policies
        if (usesRtOptLogic()) {
            SimLogger.printLine(policy + ": Starting Optimizer Loop. Delta T = " + DELTA_T);
            if (isTrainingDataActive()) {
                SimLogger.printLine("Candidate training-data output: " + TRAINING_DATA_PATH
                        + " | exploration=" + isCandidateExplorationActive()
                        + " | run_id=" + trainingRunId);
            }
            schedule(getId(), SimSettings.CLIENT_ACTIVITY_START_TIME + DELTA_T, UPDATE_OPTIMIZER);
        }
    }

    @Override
    public void shutdownEntity() {
        if (isTrainingDataActive()) {
            closeActiveTrainingWindow(CloudSim.clock());
            flushTrainingWindows(true);
        }
    }

    @Override
    public void processEvent(SimEvent ev) {
        if (ev == null) {
            SimLogger.printLine(getName() + ".processOtherEvent(): " + "Error - an event is null! Terminating simulation...");
            System.exit(1);
            return;
        }

        switch (ev.getTag()) {
            case UPDATE_PREDICTION_WINDOW: {
                statisticLogger.switchNewStatWindow();
                schedule(getId(), OrchestratorStatisticLogger.PREDICTION_WINDOW_UPDATE_INTERVAL,
                        UPDATE_PREDICTION_WINDOW);
                break;
            }
            case UPDATE_OPTIMIZER: {
                if (usesRtOptLogic()) {
                    runOptimizer();                 
                    schedule(getId(), DELTA_T, UPDATE_OPTIMIZER);
                }
                break;
            }
            default:
                SimLogger.printLine(getName() + ": unknown event type");
                break;
        }
    }

    public void processOtherEvent(SimEvent ev) {
        if (ev == null) {
            SimLogger.printLine(getName() + ".processOtherEvent(): " + "Error - an event is null! Terminating simulation...");
            System.exit(1);
            return;
        }
    }

    /**
     * Records successful task completion for learning algorithms
     *
     * @param task Completed task
     * @param serviceTime Total service time including network and processing
     * delays
     */
    public void taskCompleted(Task task, double serviceTime) {
        if (policy.equals("AI_TRAINER")) {
            trainerLogger.addSuccessStat(task, serviceTime);
        }

        if (policy.equals("PREDICTIVE")) {
            statisticLogger.addSuccessStat(task, serviceTime);
        }

        if (policy.equals("MAB")) {
            MAB.updateUCB(task, serviceTime);
        }
        if (usesRtOptLogic()) {
            int type = task.getTaskType();
            if (type >= 0 && type < NUM_APP_TYPES) {
                completedPerType[type]++;
                serviceTimeSumPerType[type] += serviceTime;
                if (serviceTime > getDeadlineForType(type)) {
                    deadlineMissesPerType[type]++;
                }
            }

            recordTrainingOutcome(task, true, serviceTime);
            taskDecisionMap.remove(task.getMobileDeviceId() + "_" + task.getCloudletId());
        }
    }

    /**
     * Records task failure for learning algorithms
     *
     * @param task Failed task
     */
    public void taskFailed(Task task) {
        if (policy.equals("AI_TRAINER")) {
            trainerLogger.addFailStat(task);
        }

        if (policy.equals("PREDICTIVE")) {
            statisticLogger.addFailStat(task);
        }

        if (policy.equals("MAB")) {
            MAB.updateUCB(task, 0);
        }

        if (usesRtOptLogic()) {
            int type = task.getTaskType();
            if (type >= 0 && type < NUM_APP_TYPES) {
                failuresPerType[type]++;
            }

            recordTrainingOutcome(task, false, 0.0);

            String key = task.getMobileDeviceId() + "_" + task.getCloudletId();
            if (taskDecisionMap.containsKey(key)) {
                int d = taskDecisionMap.get(key);
                if (d == EDGE_DATACENTER) {
                    failEdge++;
                } else if (d == CLOUD_DATACENTER_VIA_RSU) {
                    failRSU++;
                } else {
                    failGSM++;
                }
                taskDecisionMap.remove(key);
            }
        }
    }

    /**
     * Opens training data output file for AI trainer mode
     */
    public void openTrainerOutputFile() {
        trainerLogger.openTrainerOutputFile();
    }

    /**
     * Closes training data output file for AI trainer mode
     */
    public void closeTrainerOutputFile() {
        trainerLogger.closeTrainerOutputFile();
    }

    /**
     * RAIDER TRAX CORE LOGIC
     *
     * @param checkThreshold If true, returns null if server is congested.
     */
    private EdgeVM findBestEdgeServer(Task task, boolean checkThreshold) {
        double bestScore = Double.MAX_VALUE;
        EdgeVM bestVm = null;
        Location deviceLoc = SimManager.getInstance().getMobilityModel().getLocation(task.getMobileDeviceId(), CloudSim.clock());

        // SELECT PARAMETERS BASED ON TASK TYPE
        int type = task.getTaskType();
        // Fallback for unknown types to index 0
        if (type < 0 || type >= NUM_APP_TYPES) {
            type = 0;
        }

        double wLoad, wDist, threshold;

        if (usesRtOptLogic()) {
            wLoad = wLoadPerType[type];
            wDist = wDistPerType[type];
            threshold = thresholdPerType[type];
        } else {
            // Standard RAIDER_TRAX Defaults (Global)
            wLoad = 0.5;
            wDist = 0.5;
            threshold = 85.0;
        }

        List<Datacenter> datacenters = SimManager.getInstance().getEdgeServerManager().getDatacenterList();
        for (Datacenter datacenter : datacenters) {
            if (datacenter.getHostList().isEmpty()) {
                continue;
            }
            EdgeHost host = (EdgeHost) datacenter.getHostList().get(0);
            Location serverLoc = host.getLocation();
            double distance = SimUtils.calculateDistance(deviceLoc, serverLoc);

            if (host.getVmList().isEmpty()) {
                continue;
            }
            EdgeVM vm = (EdgeVM) host.getVmList().get(0);
            double currentLoad = vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock()) * 100;

            double normDist = distance / 1000.0;
            double normLoad = currentLoad / 100.0;
            double noise = simulationRandom.nextDouble() * 0.1;
            double score = (wDist * normDist) + (wLoad * normLoad) + noise;

            if (checkThreshold) {
                if (currentLoad < threshold) {
                    if (score < bestScore) {
                        bestScore = score;
                        bestVm = vm;
                    }
                }
            } else {
                if (score < bestScore) {
                    bestScore = score;
                    bestVm = vm;
                }
            }
        }
        return bestVm;
    }

    // Helper for distance calculation if not available in SimUtils for this version
    private static class SimUtils {

        public static double calculateDistance(Location loc1, Location loc2) {
            return Math.sqrt(Math.pow(loc1.getXPos() - loc2.getXPos(), 2) + Math.pow(loc1.getYPos() - loc2.getYPos(), 2));
        }
    }

    private boolean usesRtOptLogic() {
        return policy.equals("RT_OPT")
                || policy.equals("RT_OPT_ML")
                || policy.equals(EXP3_MC_POLICY)
                || policy.equals(SCOPE_DATA_COLLECTION_POLICY)
                || policy.equals(SCOPE_POLICY)
                || policy.equals(SOFTCOST_OPT_POLICY)
                || policy.equals(PERFORMANCE_ONLY_POLICY);
    }

    private boolean usesScopeOptimizerService() {
        return policy.equals("RT_OPT_ML")
                || policy.equals(SCOPE_POLICY)
                || policy.equals(SOFTCOST_OPT_POLICY)
                || policy.equals(PERFORMANCE_ONLY_POLICY);
    }

    private boolean isTrainingDataActive() {
        return policy.equals(SCOPE_DATA_COLLECTION_POLICY);
    }

    private boolean isCandidateExplorationActive() {
        return isTrainingDataActive();
    }

    private double getDeadlineForType(int type) {
        if (type < 0 || type >= NUM_APP_TYPES) {
            return 1.0;
        }
        double deadline = SimSettings.getInstance().getTaskLookUpTable()[type][13];
        return deadline > 0.0 ? deadline : 1.0;
    }

    private void recordTrainingOutcome(Task task, boolean completed, double serviceTime) {
        if (!isTrainingDataActive()) {
            return;
        }

        String key = task.getMobileDeviceId() + "_" + task.getCloudletId();
        Long windowId = taskTrainingWindowMap.remove(key);
        if (windowId == null) {
            return;
        }

        CandidateTrainingWindow trainingWindow = trainingWindows.get(windowId);
        int type = task.getTaskType();
        if (trainingWindow == null || type < 0 || type >= NUM_APP_TYPES) {
            return;
        }

        if (completed) {
            trainingWindow.completed[type]++;
            trainingWindow.serviceTimeSum[type] += serviceTime;
            if (serviceTime > getDeadlineForType(type)) {
                trainingWindow.deadlineMisses[type]++;
            }
        } else {
            trainingWindow.failed[type]++;
        }
    }

    private void closeActiveTrainingWindow(double endTime) {
        if (!isTrainingDataActive() || activeTrainingWindowId < 0) {
            return;
        }

        CandidateTrainingWindow trainingWindow = trainingWindows.get(activeTrainingWindowId);
        if (trainingWindow != null) {
            trainingWindow.closed = true;
            trainingWindow.endTime = endTime;
        }
        activeTrainingWindowId = -1;
    }

    private void startTrainingWindow(double startTime, OptimizerStateSnapshot inputState) {
        if (!isTrainingDataActive()) {
            return;
        }

        CandidateTrainingWindow trainingWindow = new CandidateTrainingWindow();
        trainingWindow.id = nextTrainingWindowId++;
        trainingWindow.startTime = startTime;
        trainingWindow.inputState = inputState;
        trainingWindow.candidate = captureCurrentConfiguration();
        trainingWindows.put(trainingWindow.id, trainingWindow);
        activeTrainingWindowId = trainingWindow.id;
    }

    private CandidateConfiguration captureCurrentConfiguration() {
        CandidateConfiguration configuration = new CandidateConfiguration();
        configuration.alpha = currentAlpha;
        System.arraycopy(wLoadPerType, 0, configuration.wLoad, 0, NUM_APP_TYPES);
        System.arraycopy(thresholdPerType, 0, configuration.threshold, 0, NUM_APP_TYPES);
        return configuration;
    }

    private OptimizerStateSnapshot captureOptimizerState(double timestamp) {
        OptimizerStateSnapshot state = new OptimizerStateSnapshot();
        state.timestamp = timestamp;
        System.arraycopy(tasksPerType, 0, state.total, 0, NUM_APP_TYPES);
        System.arraycopy(lagTasksPerType, 0, state.lagTotal, 0, NUM_APP_TYPES);
        System.arraycopy(completedPerType, 0, state.completed, 0, NUM_APP_TYPES);
        System.arraycopy(failuresPerType, 0, state.failed, 0, NUM_APP_TYPES);
        System.arraycopy(deadlineMissesPerType, 0, state.deadlineMisses, 0, NUM_APP_TYPES);

        for (int i = 0; i < NUM_APP_TYPES; i++) {
            int finalized = state.completed[i] + state.failed[i];
            state.failureRate[i] = finalized > 0
                    ? (double) state.failed[i] / finalized : 0.0;
            state.meanServiceTime[i] = state.completed[i] > 0
                    ? serviceTimeSumPerType[i] / state.completed[i] : 0.0;
            state.deadlineNormServiceTime[i] = state.completed[i] > 0
                    ? state.meanServiceTime[i] / getDeadlineForType(i)
                    : (state.failed[i] > 0 ? 1.0 : 0.0);
            state.deadlineMissRate[i] = state.completed[i] > 0
                    ? (double) state.deadlineMisses[i] / state.completed[i] : 0.0;
        }

        double[] serverLoads = getCurrentServerLoads();
        double loadSum = 0.0;
        double loadMax = 0.0;
        for (double serverLoad : serverLoads) {
            loadSum += serverLoad;
            loadMax = Math.max(loadMax, serverLoad);
        }
        state.avgEdgeCpu = serverLoads.length > 0 ? loadSum / serverLoads.length : 0.0;
        state.maxEdgeCpu = loadMax;

        state.sentEdge = sentEdge;
        state.sentRsu = sentRSU;
        state.sentGsm = sentGSM;
        state.failEdge = failEdge;
        state.failRsu = failRSU;
        state.failGsm = failGSM;
        state.edgeFailureRate = sentEdge > 0 ? (double) failEdge / sentEdge : 0.0;
        state.rsuFailureRate = sentRSU > 0 ? (double) failRSU / sentRSU : 0.0;
        state.gsmFailureRate = sentGSM > 0 ? (double) failGSM / sentGSM : 0.0;
        state.previousAlpha = currentAlpha;
        System.arraycopy(wLoadPerType, 0, state.previousWLoad, 0, NUM_APP_TYPES);
        System.arraycopy(thresholdPerType, 0, state.previousThreshold, 0, NUM_APP_TYPES);
        return state;
    }

    private double[] getCurrentServerLoads() {
        List<Datacenter> datacenters = SimManager.getInstance()
                .getEdgeServerManager().getDatacenterList();
        double[] loads = new double[datacenters.size()];
        for (int i = 0; i < datacenters.size(); i++) {
            Datacenter datacenter = datacenters.get(i);
            double load = 0.0;
            if (!datacenter.getHostList().isEmpty()) {
                EdgeHost host = (EdgeHost) datacenter.getHostList().get(0);
                if (!host.getVmList().isEmpty()) {
                    EdgeVM vm = (EdgeVM) host.getVmList().get(0);
                    double totalUtilization = vm.getCloudletScheduler()
                            .getTotalUtilizationOfCpu(CloudSim.clock());
                    if (vm.getNumberOfPes() > 0) {
                        load = (totalUtilization / vm.getNumberOfPes()) * 100.0;
                    }
                }
            }
            loads[i] = clamp(load, 0.0, 100.0);
        }
        return loads;
    }

    private void logPerformanceWindow(OptimizerStateSnapshot state) {
        StringBuilder line = new StringBuilder();
        line.append(String.format("PERF_WINDOW | Policy=%s | Time=%.1f", policy, state.timestamp));
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            line.append(String.format(
                    " | App%d Total=%d Completed=%d Failed=%d FailRate=%.4f MeanService=%.6f DeadlineNorm=%.4f DeadlineMissRate=%.4f",
                    i, state.total[i], state.completed[i], state.failed[i], state.failureRate[i],
                    state.meanServiceTime[i], state.deadlineNormServiceTime[i],
                    state.deadlineMissRate[i]));
        }
        SimLogger.printLine(line.toString());
    }

    private void resetOptimizerWindowCounters() {
        System.arraycopy(tasksPerType, 0, lagTasksPerType, 0, NUM_APP_TYPES);
        Arrays.fill(tasksPerType, 0);
        Arrays.fill(completedPerType, 0);
        Arrays.fill(failuresPerType, 0);
        Arrays.fill(deadlineMissesPerType, 0);
        Arrays.fill(serviceTimeSumPerType, 0.0);
        sentEdge = 0;
        sentRSU = 0;
        sentGSM = 0;
        failEdge = 0;
        failRSU = 0;
        failGSM = 0;
    }

    private CandidateConfiguration sampleExplorationCandidate() {
        CandidateConfiguration candidate = captureCurrentConfiguration();
        double mode = candidateRandom.nextDouble();

        // 15% no-change samples, 55% local perturbations, 30% global samples.
        // This provides both realistic neighboring actions and broad coverage.
        if (mode < 0.15) {
            return candidate;
        }

        if (mode < 0.70) {
            candidate.alpha = roundToStep(clamp(
                    currentAlpha + randomChoice(-0.10, 0.0, 0.10), 0.05, 0.95), 0.05);
            for (int i = 0; i < NUM_APP_TYPES; i++) {
                candidate.wLoad[i] = roundToStep(clamp(
                        wLoadPerType[i] + randomChoice(-0.10, 0.0, 0.10), 0.10, 0.90), 0.10);
                candidate.threshold[i] = roundToStep(clamp(
                        thresholdPerType[i] + randomChoice(-10.0, 0.0, 10.0), 20.0, 95.0), 5.0);
            }
        } else {
            candidate.alpha = (candidateRandom.nextInt(19) + 1) * 0.05;
            for (int i = 0; i < NUM_APP_TYPES; i++) {
                candidate.wLoad[i] = (candidateRandom.nextInt(9) + 1) * 0.10;
                candidate.threshold[i] = 20.0 + candidateRandom.nextInt(16) * 5.0;
            }
        }
        return candidate;
    }

    private double randomChoice(double first, double second, double third) {
        int choice = candidateRandom.nextInt(3);
        return choice == 0 ? first : (choice == 1 ? second : third);
    }

    private double roundToStep(double value, double step) {
        return Math.round(value / step) * step;
    }

    private double clamp(double value, double minimum, double maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private void applyCandidateConfiguration(CandidateConfiguration candidate) {
        currentAlpha = candidate.alpha;
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            wLoadPerType[i] = candidate.wLoad[i];
            wDistPerType[i] = 1.0 - candidate.wLoad[i];
            thresholdPerType[i] = candidate.threshold[i];
        }
    }

    private double configurationDistance(CandidateConfiguration first,
            CandidateConfiguration second) {
        double distance = Math.abs(first.alpha - second.alpha);
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            distance += Math.abs(first.wLoad[i] - second.wLoad[i]);
            distance += Math.abs(first.threshold[i] - second.threshold[i]) / 100.0;
        }
        return distance;
    }

    private void updatePreviousConfiguration() {
        prevAlpha = currentAlpha;
        System.arraycopy(wLoadPerType, 0, prevWLoad, 0, NUM_APP_TYPES);
        System.arraycopy(thresholdPerType, 0, prevThreshold, 0, NUM_APP_TYPES);
    }

    private boolean isTrainingWindowComplete(CandidateTrainingWindow trainingWindow) {
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            if (trainingWindow.completed[i] + trainingWindow.failed[i]
                    < trainingWindow.generated[i]) {
                return false;
            }
        }
        return true;
    }

    private void flushTrainingWindows(boolean includeIncomplete) {
        if (!isTrainingDataActive()) {
            return;
        }

        Iterator<Map.Entry<Long, CandidateTrainingWindow>> iterator
                = trainingWindows.entrySet().iterator();
        while (iterator.hasNext()) {
            CandidateTrainingWindow trainingWindow = iterator.next().getValue();
            if (!trainingWindow.closed) {
                continue;
            }

            boolean complete = isTrainingWindowComplete(trainingWindow);
            if (!complete && !includeIncomplete) {
                continue;
            }

            try {
                appendTrainingWindow(trainingWindow, complete);
                iterator.remove();
            } catch (IOException exception) {
                SimLogger.printLine("Training-data write failed: " + exception.getMessage());
                return;
            }
        }
    }

    private void appendTrainingWindow(CandidateTrainingWindow trainingWindow,
            boolean complete) throws IOException {
        File outputFile = new File(TRAINING_DATA_PATH);
        File parent = outputFile.getAbsoluteFile().getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("Cannot create training-data directory: " + parent);
        }
        boolean writeHeader = !outputFile.exists() || outputFile.length() == 0;

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile, true))) {
            if (writeHeader) {
                writer.println(trainingCsvHeader());
            }
            for (int appId = 0; appId < NUM_APP_TYPES; appId++) {
                writer.println(trainingCsvRow(trainingWindow, appId, complete));
            }
        }
    }

    private String trainingCsvHeader() {
        StringBuilder header = new StringBuilder();
        header.append("run_id,policy,num_devices,window_id,window_start,window_end,app_id");
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            header.append(",state_total_").append(i);
        }
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            header.append(",state_lag_total_").append(i);
        }
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            header.append(",state_completed_").append(i);
        }
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            header.append(",state_failed_").append(i);
        }
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            header.append(",state_failure_rate_").append(i);
        }
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            header.append(",state_mean_service_time_").append(i);
        }
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            header.append(",state_deadline_norm_service_time_").append(i);
        }
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            header.append(",state_deadline_miss_rate_").append(i);
        }
        header.append(",state_avg_edge_cpu,state_max_edge_cpu,state_sent_edge,state_sent_rsu,state_sent_gsm");
        header.append(",state_fail_edge,state_fail_rsu,state_fail_gsm,state_edge_failure_rate");
        header.append(",state_rsu_failure_rate,state_gsm_failure_rate,previous_alpha");
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            header.append(",previous_w_load_").append(i);
        }
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            header.append(",previous_threshold_").append(i);
        }
        header.append(",candidate_alpha");
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            header.append(",candidate_w_load_").append(i);
        }
        for (int i = 0; i < NUM_APP_TYPES; i++) {
            header.append(",candidate_threshold_").append(i);
        }
        header.append(",target_generated,target_completed,target_failed,target_finalized");
        header.append(",target_failure_rate,target_mean_service_time,target_deadline");
        header.append(",target_deadline_norm_service_time,target_deadline_misses");
        header.append(",target_deadline_miss_rate,completion_fraction,is_complete");
        return header.toString();
    }

    private String trainingCsvRow(CandidateTrainingWindow trainingWindow,
            int appId, boolean complete) {
        OptimizerStateSnapshot state = trainingWindow.inputState;
        int finalized = trainingWindow.completed[appId] + trainingWindow.failed[appId];
        double failureRate = finalized > 0
                ? (double) trainingWindow.failed[appId] / finalized : 0.0;
        double meanServiceTime = trainingWindow.completed[appId] > 0
                ? trainingWindow.serviceTimeSum[appId] / trainingWindow.completed[appId] : 0.0;
        double deadline = getDeadlineForType(appId);
        double deadlineNormServiceTime = trainingWindow.completed[appId] > 0
                ? meanServiceTime / deadline
                : (trainingWindow.failed[appId] > 0 ? 1.0 : 0.0);
        double deadlineMissRate = trainingWindow.completed[appId] > 0
                ? (double) trainingWindow.deadlineMisses[appId]
                / trainingWindow.completed[appId] : 0.0;
        double completionFraction = trainingWindow.generated[appId] > 0
                ? (double) finalized / trainingWindow.generated[appId] : 1.0;

        StringBuilder row = new StringBuilder();
        appendCsv(row, csvString(trainingRunId));
        appendCsv(row, csvString(policy));
        appendCsv(row, numOfMobileDevice);
        appendCsv(row, trainingWindow.id);
        appendCsv(row, trainingWindow.startTime);
        appendCsv(row, trainingWindow.endTime);
        appendCsv(row, appId);
        for (int value : state.total) {
            appendCsv(row, value);
        }
        for (int value : state.lagTotal) {
            appendCsv(row, value);
        }
        for (int value : state.completed) {
            appendCsv(row, value);
        }
        for (int value : state.failed) {
            appendCsv(row, value);
        }
        for (double value : state.failureRate) {
            appendCsv(row, value);
        }
        for (double value : state.meanServiceTime) {
            appendCsv(row, value);
        }
        for (double value : state.deadlineNormServiceTime) {
            appendCsv(row, value);
        }
        for (double value : state.deadlineMissRate) {
            appendCsv(row, value);
        }
        appendCsv(row, state.avgEdgeCpu);
        appendCsv(row, state.maxEdgeCpu);
        appendCsv(row, state.sentEdge);
        appendCsv(row, state.sentRsu);
        appendCsv(row, state.sentGsm);
        appendCsv(row, state.failEdge);
        appendCsv(row, state.failRsu);
        appendCsv(row, state.failGsm);
        appendCsv(row, state.edgeFailureRate);
        appendCsv(row, state.rsuFailureRate);
        appendCsv(row, state.gsmFailureRate);
        appendCsv(row, state.previousAlpha);
        for (double value : state.previousWLoad) {
            appendCsv(row, value);
        }
        for (double value : state.previousThreshold) {
            appendCsv(row, value);
        }
        appendCsv(row, trainingWindow.candidate.alpha);
        for (double value : trainingWindow.candidate.wLoad) {
            appendCsv(row, value);
        }
        for (double value : trainingWindow.candidate.threshold) {
            appendCsv(row, value);
        }
        appendCsv(row, trainingWindow.generated[appId]);
        appendCsv(row, trainingWindow.completed[appId]);
        appendCsv(row, trainingWindow.failed[appId]);
        appendCsv(row, finalized);
        appendCsv(row, failureRate);
        appendCsv(row, meanServiceTime);
        appendCsv(row, deadline);
        appendCsv(row, deadlineNormServiceTime);
        appendCsv(row, trainingWindow.deadlineMisses[appId]);
        appendCsv(row, deadlineMissRate);
        appendCsv(row, completionFraction);
        appendCsv(row, complete ? 1 : 0);
        return row.toString();
    }

    private void appendCsv(StringBuilder row, Object value) {
        if (row.length() > 0) {
            row.append(',');
        }
        row.append(value);
    }

    private String csvString(String value) {
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }

    /**
     * RT_OPT: Calls the Python Optimizer API
     */
    private void runOptimizer() {
        final double timestamp = CloudSim.clock();
        closeActiveTrainingWindow(timestamp);
        flushTrainingWindows(false);

        final OptimizerStateSnapshot inputState = captureOptimizerState(timestamp);
        // logPerformanceWindow(inputState);

        try {
            CandidateConfiguration beforeUpdate = captureCurrentConfiguration();

            if (isCandidateExplorationActive()) {
                CandidateConfiguration sampledCandidate = sampleExplorationCandidate();
                applyCandidateConfiguration(sampledCandidate);
                double stepChurn = configurationDistance(beforeUpdate, sampledCandidate);
                totalChurnAccumulated += stepChurn;
                updatePreviousConfiguration();
                SimLogger.printLine(String.format(
                        "TRAINING_CANDIDATE | Policy=%s | Time=%.1f | StepChurn=%.4f | Alpha=%.2f",
                        policy, timestamp, stepChurn, currentAlpha));
            } else {

                String targetUrl;
                if (usesScopeOptimizerService()) {
                    targetUrl = RT_OPT_ML_URL;
                } else if (policy.equals(EXP3_MC_POLICY)) {
                    targetUrl = EXP3_MC_URL;
                } else {
                    targetUrl = OPTIMIZER_URL;
                }

                StringBuilder json = new StringBuilder();
                json.append("{");
                json.append("\"timestamp\":").append(timestamp).append(",");
                json.append("\"num_devices\":").append(numOfMobileDevice).append(",");

                json.append("\"policy\":\"").append(policy).append("\",");
                json.append("\"run_id\":\"").append(trainingRunId).append("\",");
                json.append("\"simulation_seed\":")
                        .append(System.getProperty("scope.simSeed", "20260828"))
                        .append(",");
                json.append("\"iteration\":")
                        .append(System.getProperty(
                                "scope.iterationNumber", "-1"))
                        .append(",");
                json.append("\"path_stats\":{");
                json.append("\"sent_edge\":").append(inputState.sentEdge).append(",");
                json.append("\"sent_rsu\":").append(inputState.sentRsu).append(",");
                json.append("\"sent_gsm\":").append(inputState.sentGsm).append(",");
                json.append("\"fail_edge\":").append(inputState.failEdge).append(",");
                json.append("\"fail_rsu\":").append(inputState.failRsu).append(",");
                json.append("\"fail_gsm\":").append(inputState.failGsm).append(",");
                json.append("\"curr_alpha\":").append(inputState.previousAlpha);
                json.append("},");

                json.append("\"app_stats\":{");

                for (int i = 0; i < NUM_APP_TYPES; i++) {
                    int finalized = inputState.completed[i] + inputState.failed[i];
                    int activeTasks = Math.max(0, inputState.total[i] - finalized);
                    json.append("\"").append(i).append("\":{");
                    json.append("\"failures\":").append(inputState.failed[i]).append(",");
                    json.append("\"completed\":").append(inputState.completed[i]).append(",");
                    json.append("\"finalized\":").append(finalized).append(",");
                    json.append("\"active_tasks\":").append(activeTasks).append(",");
                    json.append("\"active_task\":").append(activeTasks).append(",");
                    json.append("\"total\":").append(inputState.total[i]).append(",");
                    json.append("\"lag\":").append(inputState.lagTotal[i]).append(",");
                    json.append("\"failure_rate\":").append(inputState.failureRate[i]).append(",");
                    json.append("\"mean_service_time\":").append(inputState.meanServiceTime[i]).append(",");
                    json.append("\"deadline_norm_service_time\":")
                            .append(inputState.deadlineNormServiceTime[i]).append(",");
                    json.append("\"deadline_misses\":").append(inputState.deadlineMisses[i]).append(",");
                    json.append("\"deadline_miss_rate\":").append(inputState.deadlineMissRate[i]).append(",");
                    json.append("\"deadline\":").append(getDeadlineForType(i)).append(",");
                    json.append("\"curr_w_load\":").append(inputState.previousWLoad[i]).append(",");
                    json.append("\"curr_w_dist\":").append(1.0 - inputState.previousWLoad[i]).append(",");
                    json.append("\"curr_thr\":").append(inputState.previousThreshold[i]).append(",");
                    json.append("\"threshold\":").append(inputState.previousThreshold[i]).append("}");
                    if (i < NUM_APP_TYPES - 1) {
                        json.append(",");
                    }
                }
                json.append("},");

                double[] serverLoads = getCurrentServerLoads();
                json.append("\"servers\":[");
                for (int i = 0; i < serverLoads.length; i++) {
                    json.append("{\"load\":").append(serverLoads[i]).append("}");
                    if (i < serverLoads.length - 1) {
                        json.append(",");
                    }
                }
                json.append("]}");

                URL url = new URL(targetUrl);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setRequestProperty("Content-Type", "application/json; utf-8");
                connection.setDoOutput(true);
                try (OutputStream output = connection.getOutputStream()) {
                    output.write(json.toString().getBytes("utf-8"));
                }

                String responseLine;
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(connection.getInputStream(), "utf-8"))) {
                    responseLine = reader.readLine();
                } finally {
                    connection.disconnect();
                }
                if (responseLine == null) {
                    throw new IOException("Optimizer returned an empty response");
                }

                String response = responseLine.replace("{", "")
                        .replace("}", "").replace("\"", "");
                String[] sections = response.split("\\|");
                for (String section : sections) {
                    if (section.startsWith("ALPHA")) {
                        currentAlpha = Double.parseDouble(section.split(":")[1]);
                    } else if (section.contains(":")) {
                        String[] parts = section.split(":");
                        if (parts.length == 4) {
                            int id = Integer.parseInt(parts[0]);
                            if (id >= 0 && id < NUM_APP_TYPES) {
                                wLoadPerType[id] = Double.parseDouble(parts[1]);
                                wDistPerType[id] = Double.parseDouble(parts[2]);
                                thresholdPerType[id] = Double.parseDouble(parts[3]);
                            }
                        }
                    }
                }

                CandidateConfiguration afterUpdate = captureCurrentConfiguration();
                double stepChurn = configurationDistance(beforeUpdate, afterUpdate);
                totalChurnAccumulated += stepChurn;
                updatePreviousConfiguration();
//                SimLogger.printLine(String.format(
//                        "OPT_UPDATE | Policy=%s | Time=%.1f | StepChurn=%.4f | TotalChurn=%.4f | Alpha=%.2f",
//                        policy, timestamp, stepChurn, totalChurnAccumulated, currentAlpha));
            }

        } catch (Exception e) {
            SimLogger.printLine(policy + " Error: " + e.getMessage());
        } finally {
            startTrainingWindow(timestamp, inputState);
            resetOptimizerWindowCounters();
        }
    }

}
