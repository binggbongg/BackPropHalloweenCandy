using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace BackPropHalloweenCandy
{
    internal class CSVLoader
    {
        public const int NumFeatures = 11; // from chocolate to pricepercent
        public const int NumClasses = 3;   // Low, Medium, High popularity
        private const int NumCsvColumns = 13;

        public static void Load(string path, out double[][] features,
            out double[] winPercents, out string[] names)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException("CSV file not found: " + path);

            List<double[]> featureList = new List<double[]>();
            List<double> winList = new List<double>();
            List<string> nameList = new List<string>();
            int skipped = 0;

            string[] lines = File.ReadAllLines(path);
            for (int i = 1; i < lines.Length; ++i) // line 0 is the header
            {
                if (string.IsNullOrWhiteSpace(lines[i]))
                    continue;

                string[] parts = lines[i].Split(',');
                if (parts.Length != NumCsvColumns)
                {
                    ++skipped;
                    continue;
                }

                // columns 1 to 11 are features (column 0 = name, column 12 = winpercent)
                double[] row = new double[NumFeatures];
                bool ok = true;
                for (int j = 0; j < NumFeatures; ++j)
                {
                    double v;
                    if (!double.TryParse(parts[j + 1].Trim(), NumberStyles.Float,
                        CultureInfo.InvariantCulture, out v))
                    {
                        ok = false;
                        break;
                    }
                    row[j] = v;
                }

                double win = 0.0;
                if (!ok || !double.TryParse(parts[12].Trim(), NumberStyles.Float,
                    CultureInfo.InvariantCulture, out win))
                {
                    ++skipped;
                    continue;
                }

                nameList.Add(parts[0].Trim());
                featureList.Add(row);
                winList.Add(win);
            }

            if (skipped > 0)
                Console.WriteLine("Warning: skipped " + skipped + " bad/incomplete row(s)");

            features = featureList.ToArray();
            winPercents = winList.ToArray();
            names = nameList.ToArray();
        }

        /// Converts winpercent into 3 equal-sized popularity tiers (tertiles) and
        /// builds the full training rows [features | one-hot tier].
        ///   tier 0 = Low    (winpercent below lowCut)
        ///   tier 1 = Medium (lowCut up to highCut)
        ///   tier 2 = High   (highCut and above)
        /// </summary>
        public static double[][] BuildTierRows(double[][] features, double[] winPercents,
            out double lowCut, out double highCut)
        {
            int n = features.Length;

            double[] sorted = (double[])winPercents.Clone();
            Array.Sort(sorted);
            lowCut = sorted[n / 3];          // ~33rd percentile
            highCut = sorted[(2 * n) / 3];   // ~67th percentile

            double[][] rows = new double[n][];
            for (int r = 0; r < n; ++r)
            {
                rows[r] = new double[NumFeatures + NumClasses];
                Array.Copy(features[r], rows[r], NumFeatures);

                int tier;
                if (winPercents[r] < lowCut) tier = 0;
                else if (winPercents[r] < highCut) tier = 1;
                else tier = 2;

                rows[r][NumFeatures + tier] = 1.0; // one-hot target
            }
            return rows;
        }
    }
}
