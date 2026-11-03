using System;

namespace BackPropHalloweenCandy
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin neural network back-propagation demo");

            int numInput = CSVLoader.NumFeatures; // number features (11 candy attributes)
            int numHidden = 5;
            int numOutput = CSVLoader.NumClasses; // number of classes for Y (Low, Medium, High)
            int seed = 1; 


            // (candy-data.csv must be set to "Copy to Output Directory: Copy if newer")
            Console.WriteLine("\nLoading candy data from candy-data.csv");
            double[][] features;
            double[] winPercents;
            string[] names;
            CSVLoader.Load("candy-data.csv", out features, out winPercents, out names);

            double lowCut, highCut;
            double[][] allData = CSVLoader.BuildTierRows(features, winPercents,
              out lowCut, out highCut);
            Console.WriteLine("Loaded " + allData.Length + " candies with " + numInput + " features");
            Console.WriteLine("Tiers by winpercent: Low < " + lowCut.ToString("F2") +
              " <= Medium < " + highCut.ToString("F2") + " <= High");
            Console.WriteLine("Done");

            Console.WriteLine("\nCreating train (80%) and test (20%) matrices");
            double[][] trainData;
            double[][] testData;
            SplitTrainTest(allData, 0.80, seed, out trainData, out testData);
            Console.WriteLine("Done\n");

            // ADDED: min-max normalize inputs, fitted on training data only
            Console.WriteLine("Normalizing features to [0,1] (fit on training data only)");
            Normalizer normalizer = new Normalizer(numInput);
            normalizer.Fit(trainData);
            normalizer.Apply(trainData);
            normalizer.Apply(testData);
            Console.WriteLine("Done\n");

            Console.WriteLine("Training data:");
            ShowMatrix(trainData, 4, 2, true);
            Console.WriteLine("Test data:");
            ShowMatrix(testData, 4, 2, true);

            Console.WriteLine("Creating a " + numInput + "-" + numHidden +
              "-" + numOutput + " neural network");
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

            int maxEpochs = 1000;
            double learnRate = 0.05;
            double momentum = 0.01;
            bool useCrossEntropy = true; // set to false to run initial ver
            double weightDecay = 0.01; // set to 0.0 to run initial ver
            int logInterval = 100; // ADDED: log loss every 100 epochs
            string logPath = "training_log.csv"; // ADDED: loss log file
            Console.WriteLine("\nSetting maxEpochs = " + maxEpochs);
            Console.WriteLine("Setting learnRate = " + learnRate.ToString("F2"));
            Console.WriteLine("Setting momentum  = " + momentum.ToString("F2"));
            Console.WriteLine("Using cross entropy = " + useCrossEntropy);
            Console.WriteLine("Setting weightDecay  = " + weightDecay);

            Console.WriteLine("\nStarting training");
            // Train now also takes testData, logInterval, and logPath
            double[] weights = nn.Train(trainData, testData, maxEpochs, learnRate, momentum,
              useCrossEntropy, weightDecay, logInterval, logPath);
            Console.WriteLine("Done");
            Console.WriteLine("\nFinal neural network model weights and biases:\n");
            ShowVector(weights, 2, 10, true);

            double trainAcc = nn.Accuracy(trainData);
            Console.WriteLine("\nFinal accuracy on training data = " +
              trainAcc.ToString("F4"));

            double testAcc = nn.Accuracy(testData);
            Console.WriteLine("Final accuracy on test data     = " +
              testAcc.ToString("F4"));

            Console.WriteLine("\nEnd back-propagation demo\n");
            Console.ReadLine();
        } // Main

        public static void ShowMatrix(double[][] matrix, int numRows,
          int decimals, bool indices)
        {
            int len = matrix.Length.ToString().Length;
            for (int i = 0; i < numRows; ++i)
            {
                if (indices == true)
                    Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    double v = matrix[i][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
                Console.WriteLine("");
            }

            if (numRows < matrix.Length)
            {
                Console.WriteLine(". . .");
                int lastRow = matrix.Length - 1;
                if (indices == true)
                    Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[lastRow].Length; ++j)
                {
                    double v = matrix[lastRow][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
            }
            Console.WriteLine("\n");
        }

        public static void ShowVector(double[] vector, int decimals,
          int lineLen, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % lineLen == 0) Console.WriteLine("");
                if (vector[i] >= 0) Console.Write(" ");
                Console.Write(vector[i].ToString("F" + decimals) + " ");
            }
            if (newLine == true)
                Console.WriteLine("");
        }

        static void SplitTrainTest(double[][] allData, double trainPct,
          int seed, out double[][] trainData, out double[][] testData)
        {
            Random rnd = new Random(seed);
            int totRows = allData.Length;
            int numTrainRows = (int)(totRows * trainPct); // usually 0.80
            int numTestRows = totRows - numTrainRows;
            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];

            double[][] copy = new double[allData.Length][]; // ref copy of data
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i) // scramble order
            {
                int r = rnd.Next(i, copy.Length); // use Fisher-Yates
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (int i = 0; i < numTrainRows; ++i)
                trainData[i] = copy[i];

            for (int i = 0; i < numTestRows; ++i)
                testData[i] = copy[i + numTrainRows];
        } // SplitTrainTest

    } // Program
}