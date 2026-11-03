using System;
using System.Collections.Generic;

namespace BackPropHalloweenCandy
{
    internal class Normalizer
    {
        private readonly int numCols; // number of input columns to scale
        private double[] mins;
        private double[] maxs;

        public Normalizer(int numCols)
        {
            this.numCols = numCols;
            this.mins = new double[numCols];
            this.maxs = new double[numCols];
        }

        /// <summary>Learn per-column min and max from the training data.</summary>
        public void Fit(double[][] trainData)
        {
            for (int j = 0; j < numCols; ++j)
            {
                mins[j] = double.MaxValue;
                maxs[j] = double.MinValue;
                for (int i = 0; i < trainData.Length; ++i)
                {
                    double v = trainData[i][j];
                    if (v < mins[j]) mins[j] = v;
                    if (v > maxs[j]) maxs[j] = v;
                }
            }
        }

        /// <summary>Scale the input columns of data in place using the fitted min/max.</summary>
        public void Apply(double[][] data)
        {
            for (int i = 0; i < data.Length; ++i)
            {
                for (int j = 0; j < numCols; ++j)
                {
                    double range = maxs[j] - mins[j];
                    if (range == 0.0)
                        data[i][j] = 0.0; // constant column: avoid divide-by-zero
                    else
                        data[i][j] = (data[i][j] - mins[j]) / range;
                }
            }
        }

        /// <summary>Scale a single row of input features using the fitted min/max.</summary>
        public double[] ApplySingle(double[] row)
        {
            double[] normalized = new double[numCols];
            for (int j = 0; j < numCols; ++j)
            {
                double range = maxs[j] - mins[j];
                if (range == 0.0)
                    normalized[j] = 0.0;
                else
                    normalized[j] = (row[j] - mins[j]) / range;
            }
            return normalized;
        }
    }
}
