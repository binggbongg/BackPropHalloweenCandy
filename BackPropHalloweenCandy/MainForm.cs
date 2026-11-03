using System;
using System.Drawing;
using System.Windows.Forms;
using System.Threading.Tasks;

namespace BackPropHalloweenCandy
{
    public partial class MainForm : Form
    {
        // Model state
        private NeuralNetwork? _nn;
        private Normalizer?    _normalizer;
        private double[][]?    _trainData;
        private double[][]?    _testData;
        private string[]?      _trainNames;
        private string[]?      _testNames;
        private double         _lowCut;
        private double         _highCut;

        private static readonly string[] FeatureNames = new[]
        {
            "Chocolate", "Fruity", "Caramel", "Peanuty / Almondy",
            "Nougat", "Crisped Rice Wafer", "Hard Candy",
            "Bar Shape", "Pluribus (Multi-piece)", "Sugar Percentile", "Price Percentile"
        };

        private static readonly string[] TierNames = { "Low", "Medium", "High" };

        // Controls
        private TabControl tabControl = null!;
        private TabPage tabTrain = null!, tabTestPreds = null!,
                        tabTrainPreds = null!, tabMatrix = null!, tabPredict = null!;

        // Training tab
        private RichTextBox rtbLog = null!;
        private Button btnTrain = null!;
        private ProgressBar pbTraining = null!;
        private Label lblStatus = null!;
        private NumericUpDown nudEpochs = null!, nudHidden = null!;
        private NumericUpDown nudLearnRate = null!, nudMomentum = null!, nudWeightDecay = null!;
        private CheckBox chkCrossEntropy = null!;

        // Predictions tabs
        private DataGridView dgvTestPreds = null!, dgvTrainPreds = null!;
        private Label lblTestSummary = null!, lblTrainSummary = null!;

        // Confusion matrix tab
        private DataGridView dgvMatrix = null!;
        private RichTextBox rtbMetrics = null!;

        // Predict custom tab
        private CheckBox[] chkBinaryFeatures = null!;
        private TrackBar[] trkContinuous = null!;
        private Label[] lblContinuousVal = null!;
        private Button btnPredict = null!;
        private Label lblPredictedTier = null!, lblConfidence = null!;
        private ProbabilityBarPanel probBars = null!;

        public MainForm()
        {
            InitializeComponent();
            BuildUI();
        }

        private void BuildUI()
        {
            this.Text = "Halloween Candy Popularity Predictor - Backpropagation Demo";
            this.Size = new Size(1000, 680);
            this.MinimumSize = new Size(850, 580);
            this.StartPosition = FormStartPosition.CenterScreen;

            tabControl = new TabControl { Dock = DockStyle.Fill };

            tabTrain      = new TabPage("Training");
            tabTestPreds  = new TabPage("Test Predictions");
            tabTrainPreds = new TabPage("Train Predictions");
            tabMatrix     = new TabPage("Confusion Matrix");
            tabPredict    = new TabPage("Predict Custom Candy");

            tabControl.TabPages.AddRange(new[] { tabTrain, tabTestPreds, tabTrainPreds, tabMatrix, tabPredict });

            BuildTrainingTab();
            BuildPredictionsTab(tabTestPreds,  out dgvTestPreds,  out lblTestSummary);
            BuildPredictionsTab(tabTrainPreds, out dgvTrainPreds, out lblTrainSummary);
            BuildConfusionMatrixTab();
            BuildCustomCandyTab();

            this.Controls.Add(tabControl);

            SetResultTabsEnabled(false);
        }

        // Training Tab
        private void BuildTrainingTab()
        {
            var pnlOuter = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2,
                RowCount = 1
            };
            pnlOuter.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 250));
            pnlOuter.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

            // Left: Hyperparameter group box
            var grpHyper = new GroupBox
            {
                Text = "Hyperparameters",
                Dock = DockStyle.Fill,
                Padding = new Padding(8)
            };

            int y = 20;
            nudEpochs     = AddLabeledNumeric(grpHyper, "Max Epochs:",   ref y, 1,     50000, 1000,  0);
            nudHidden     = AddLabeledNumeric(grpHyper, "Hidden Nodes:", ref y, 2,     100,   5,     0);
            nudLearnRate  = AddLabeledNumeric(grpHyper, "Learn Rate:",   ref y, 0.001m, 1m,  0.05m, 4);
            nudMomentum   = AddLabeledNumeric(grpHyper, "Momentum:",     ref y, 0m,     1m,  0.01m, 4);
            nudWeightDecay= AddLabeledNumeric(grpHyper, "Weight Decay:", ref y, 0m,     1m,  0.01m, 4);

            chkCrossEntropy = new CheckBox
            {
                Text = "Use Cross-Entropy Loss",
                Checked = true,
                Location = new Point(10, y),
                AutoSize = true
            };
            grpHyper.Controls.Add(chkCrossEntropy);
            y += 30;

            y += 6;
            btnTrain = new Button
            {
                Text = "Train Network",
                Location = new Point(10, y),
                Size = new Size(210, 28)
            };
            btnTrain.Click += BtnTrain_Click;
            grpHyper.Controls.Add(btnTrain);
            y += 36;

            pbTraining = new ProgressBar
            {
                Location = new Point(10, y),
                Size = new Size(210, 16),
                Style = ProgressBarStyle.Blocks,
                Minimum = 0,
                Maximum = 100,
                Value = 0
            };
            grpHyper.Controls.Add(pbTraining);
            y += 24;

            lblStatus = new Label
            {
                Text = "Ready. Click \"Train Network\" to begin.",
                Location = new Point(10, y),
                Size = new Size(220, 36),
                AutoSize = false
            };
            grpHyper.Controls.Add(lblStatus);

            // Right: Log group box
            var grpLog = new GroupBox
            {
                Text = "Training Log",
                Dock = DockStyle.Fill
            };

            rtbLog = new RichTextBox
            {
                Dock = DockStyle.Fill,
                ReadOnly = true,
                Font = new Font("Courier New", 9f),
                BorderStyle = BorderStyle.None
            };

            grpLog.Controls.Add(rtbLog);

            pnlOuter.Controls.Add(grpHyper, 0, 0);
            pnlOuter.Controls.Add(grpLog,   1, 0);
            tabTrain.Controls.Add(pnlOuter);
        }

        private NumericUpDown AddLabeledNumeric(GroupBox parent, string labelText,
            ref int y, decimal min, decimal max, decimal val, int decimals)
        {
            parent.Controls.Add(new Label
            {
                Text = labelText,
                Location = new Point(10, y),
                AutoSize = true
            });
            y += 20;

            var nud = new NumericUpDown
            {
                Location = new Point(10, y),
                Size = new Size(210, 23),
                Minimum = min,
                Maximum = max,
                Value = val,
                DecimalPlaces = decimals,
                Increment = decimals == 0 ? 1 : (decimal)Math.Pow(10, -decimals)
            };
            parent.Controls.Add(nud);
            y += 30;
            return nud;
        }

        // Predictions Tab
        private void BuildPredictionsTab(TabPage tab, out DataGridView dgv, out Label summaryLabel)
        {
            var pnl = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                RowCount = 2,
                ColumnCount = 1
            };
            pnl.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            pnl.RowStyles.Add(new RowStyle(SizeType.Absolute, 30));

            dgv = new DataGridView
            {
                Dock = DockStyle.Fill,
                ReadOnly = true,
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false,
                AllowUserToResizeRows = false,
                RowHeadersVisible = false,
                SelectionMode = DataGridViewSelectionMode.FullRowSelect,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill
            };

            var leftStyle = new DataGridViewCellStyle { Alignment = DataGridViewContentAlignment.MiddleLeft };
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "Name",      HeaderText = "Candy Name",  FillWeight = 25, DefaultCellStyle = leftStyle });
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "Actual",    HeaderText = "Actual Tier", FillWeight = 10 });
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "Predicted", HeaderText = "Predicted",   FillWeight = 10 });
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "ProbLow",   HeaderText = "P(Low)",      FillWeight = 10 });
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "ProbMed",   HeaderText = "P(Medium)",   FillWeight = 10 });
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "ProbHigh",  HeaderText = "P(High)",     FillWeight = 10 });
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "Result",    HeaderText = "Result",      FillWeight = 7  });
            dgv.CellFormatting += Dgv_CellFormatting;

            summaryLabel = new Label
            {
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft,
                Padding = new Padding(4, 0, 0, 0)
            };

            pnl.Controls.Add(dgv, 0, 0);
            pnl.Controls.Add(summaryLabel, 0, 1);
            tab.Controls.Add(pnl);
        }

        // Confusion Matrix Tab
        private void BuildConfusionMatrixTab()
        {
            var split = new SplitContainer
            {
                Dock = DockStyle.Fill,
                Orientation = Orientation.Horizontal
            };

            dgvMatrix = new DataGridView
            {
                Dock = DockStyle.Fill,
                ReadOnly = true,
                AllowUserToAddRows = false,
                AllowUserToResizeRows = false,
                RowHeadersVisible = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.AllCells
            };

            rtbMetrics = new RichTextBox
            {
                Dock = DockStyle.Fill,
                ReadOnly = true,
                Font = new Font("Courier New", 9f)
            };

            split.Panel1.Controls.Add(dgvMatrix);
            split.Panel2.Controls.Add(rtbMetrics);
            split.SplitterDistance = 180;

            tabMatrix.Controls.Add(split);
        }

        // Custom Candy Tab
        private void BuildCustomCandyTab()
        {
            chkBinaryFeatures = new CheckBox[9];
            trkContinuous     = new TrackBar[2];
            lblContinuousVal  = new Label[2];

            // Left scroll panel
            var scrollPanel = new Panel
            {
                Dock = DockStyle.Left,
                Width = 350,
                AutoScroll = true
            };

            // Right results panel
            var pnlRight = new Panel
            {
                Dock = DockStyle.Fill,
                Padding = new Padding(12)
            };

            int y = 10;

            var grpBinary = new GroupBox
            {
                Text = "Candy Attributes (Yes / No)",
                Location = new Point(8, y),
                Size = new Size(330, 270),
                AutoSize = false
            };

            int gy = 20;
            for (int i = 0; i < 9; i++)
            {
                chkBinaryFeatures[i] = new CheckBox
                {
                    Text = FeatureNames[i],
                    Location = new Point(10, gy),
                    AutoSize = true
                };
                grpBinary.Controls.Add(chkBinaryFeatures[i]);
                gy += 26;
            }

            scrollPanel.Controls.Add(grpBinary);
            y += grpBinary.Height + 10;

            var grpCont = new GroupBox
            {
                Text = "Continuous Attributes",
                Location = new Point(8, y),
                Size = new Size(330, 120)
            };

            string[] contLabels = { "Sugar Percentile (0.0 - 1.0)", "Price Percentile (0.0 - 1.0)" };
            int cy = 18;
            for (int i = 0; i < 2; i++)
            {
                int capturedI = i;

                grpCont.Controls.Add(new Label
                {
                    Text = contLabels[i],
                    Location = new Point(10, cy),
                    AutoSize = true
                });

                lblContinuousVal[i] = new Label
                {
                    Text = "0.50",
                    Location = new Point(285, cy),
                    AutoSize = true
                };
                grpCont.Controls.Add(lblContinuousVal[i]);
                cy += 18;

                trkContinuous[i] = new TrackBar
                {
                    Location = new Point(6, cy),
                    Size = new Size(315, 36),
                    Minimum = 0,
                    Maximum = 100,
                    Value = 50,
                    TickFrequency = 10,
                    SmallChange = 1,
                    LargeChange = 10
                };
                trkContinuous[i].ValueChanged += (s, e) =>
                {
                    lblContinuousVal[capturedI].Text =
                        (trkContinuous[capturedI].Value / 100.0).ToString("F2");
                };
                grpCont.Controls.Add(trkContinuous[i]);
                cy += 40;
            }

            scrollPanel.Controls.Add(grpCont);
            y += grpCont.Height + 10;

            btnPredict = new Button
            {
                Text = "Predict Likelihood Tier",
                Location = new Point(8, y),
                Size = new Size(330, 28),
                Enabled = false
            };
            btnPredict.Click += BtnPredict_Click;
            scrollPanel.Controls.Add(btnPredict);

            // Right side: results
            lblPredictedTier = new Label
            {
                Text = "Train the network first, then predict.",
                Font = new Font(this.Font, FontStyle.Bold),
                TextAlign = ContentAlignment.TopCenter,
                Dock = DockStyle.Top,
                Height = 40,
                Padding = new Padding(0, 10, 0, 0)
            };

            lblConfidence = new Label
            {
                Text = "",
                TextAlign = ContentAlignment.TopCenter,
                Dock = DockStyle.Top,
                Height = 24
            };

            probBars = new ProbabilityBarPanel
            {
                Dock = DockStyle.Top,
                Height = 100
            };

            pnlRight.Controls.Add(probBars);
            pnlRight.Controls.Add(lblConfidence);
            pnlRight.Controls.Add(lblPredictedTier);

            tabPredict.Controls.Add(pnlRight);
            tabPredict.Controls.Add(scrollPanel);
        }

        // Training Logic
        private async void BtnTrain_Click(object? sender, EventArgs e)
        {
            btnTrain.Enabled = false;
            pbTraining.Value = 0;
            lblStatus.Text = "Training in progress...";
            SetResultTabsEnabled(false);
            rtbLog.Clear();

            int    maxEpochs   = (int)nudEpochs.Value;
            int    numHidden   = (int)nudHidden.Value;
            double learnRate   = (double)nudLearnRate.Value;
            double momentum    = (double)nudMomentum.Value;
            double weightDecay = (double)nudWeightDecay.Value;
            bool   crossEntropy = chkCrossEntropy.Checked;

            Log("Loading candy-data.csv...");

            try
            {
                await Task.Run(() =>
                {
                    int numInput  = CSVLoader.NumFeatures;
                    int numOutput = CSVLoader.NumClasses;

                    CSVLoader.Load("candy-data.csv",
                        out double[][] features, out double[] winPercents, out string[] names);

                    double[][] allData = CSVLoader.BuildTierRows(features, winPercents,
                        out double lowCut, out double highCut);
                    _lowCut  = lowCut;
                    _highCut = highCut;

                    AppendLog($"Loaded {allData.Length} candies with {numInput} features.");
                    AppendLog($"Tiers: Low < {lowCut:F2}% | Medium {lowCut:F2}%-{highCut:F2}% | High >= {highCut:F2}%");

                    SplitTrainTest(allData, names, 0.80, 1,
                        out double[][] trainData, out double[][] testData,
                        out string[] trainNames, out string[] testNames);

                    AppendLog($"Split: {trainData.Length} train / {testData.Length} test.");

                    var normalizer = new Normalizer(numInput);
                    normalizer.Fit(trainData);
                    normalizer.Apply(trainData);
                    normalizer.Apply(testData);
                    AppendLog("Features normalized [0,1] on training data.");

                    var nn = new NeuralNetwork(numInput, numHidden, numOutput);
                    AppendLog($"Network: {numInput}-{numHidden}-{numOutput} | " +
                              $"epochs={maxEpochs} lr={learnRate:F4} mom={momentum:F4} " +
                              $"wd={weightDecay:F4} cross-entropy={crossEntropy}");
                    AppendLog("Training...");

                    nn.Train(trainData, testData, maxEpochs, learnRate, momentum,
                             crossEntropy, weightDecay, 100, "training_log.csv",
                             (epoch, trainLoss, testLoss) =>
                             {
                                 int pct = (int)((double)epoch / maxEpochs * 100);
                                 if (pct > 99) pct = 99;
                                 this.BeginInvoke(() =>
                                 {
                                     pbTraining.Value = pct;
                                     lblStatus.Text = $"Training: epoch {epoch}/{maxEpochs}";
                                 });
                             });

                    double trainAcc = nn.Accuracy(trainData);
                    double testAcc  = nn.Accuracy(testData);
                    AppendLog($"Done!  Train accuracy = {trainAcc:P2}  |  Test accuracy = {testAcc:P2}");

                    _nn         = nn;
                    _normalizer = normalizer;
                    _trainData  = trainData;
                    _testData   = testData;
                    _trainNames = trainNames;
                    _testNames  = testNames;
                });

                pbTraining.Value = 100;
                PopulatePredictionsGrid(dgvTestPreds,  lblTestSummary,  _testData!,  _testNames!);
                PopulatePredictionsGrid(dgvTrainPreds, lblTrainSummary, _trainData!, _trainNames!);
                PopulateConfusionMatrix();
                SetResultTabsEnabled(true);
                btnPredict.Enabled = true;
                lblPredictedTier.Text = "Enter a recipe on the left and click Predict.";
                lblStatus.Text = "Training complete.";
            }
            catch (Exception ex)
            {
                AppendLog($"ERROR: {ex.Message}");
                lblStatus.Text = "Error - see log.";
            }

            btnTrain.Enabled = true;
        }

        // Populate Results
        private void PopulatePredictionsGrid(DataGridView dgv, Label summary,
            double[][] data, string[] names)
        {
            dgv.Rows.Clear();
            int correct     = 0;
            int numFeatures = CSVLoader.NumFeatures;
            int numClasses  = CSVLoader.NumClasses;

            for (int i = 0; i < data.Length; i++)
            {
                double[] x = new double[numFeatures];
                Array.Copy(data[i], x, numFeatures);

                double[] probs = _nn!.ComputeOutputs(x);
                int pred = ArgMax(probs);

                int actual = -1;
                for (int c = 0; c < numClasses; c++)
                    if (data[i][numFeatures + c] == 1.0) { actual = c; break; }

                bool ok = pred == actual;
                if (ok) correct++;

                dgv.Rows.Add(
                    names.Length > i ? names[i] : $"Candy #{i + 1}",
                    TierNames[actual],
                    TierNames[pred],
                    $"{probs[0]:P1}",
                    $"{probs[1]:P1}",
                    $"{probs[2]:P1}",
                    ok ? "OK" : "MISS");

                dgv.Rows[i].Tag = ok;
            }

            double acc = (double)correct / data.Length;
            summary.Text = $"  {correct} / {data.Length} correct  —  Accuracy: {acc:P2}";
        }

        private void PopulateConfusionMatrix()
        {
            int[,] mat = new int[3, 3];
            int numFeatures = CSVLoader.NumFeatures;

            foreach (double[] row in _testData!)
            {
                double[] x = new double[numFeatures];
                Array.Copy(row, x, numFeatures);
                double[] probs = _nn!.ComputeOutputs(x);
                int pred   = ArgMax(probs);
                int actual = 0;
                for (int c = 0; c < 3; c++)
                    if (row[numFeatures + c] == 1.0) { actual = c; break; }
                mat[actual, pred]++;
            }

            dgvMatrix.Rows.Clear();
            dgvMatrix.Columns.Clear();
            dgvMatrix.Columns.Add("row",   "");
            dgvMatrix.Columns.Add("pLow",  "Pred: Low");
            dgvMatrix.Columns.Add("pMed",  "Pred: Medium");
            dgvMatrix.Columns.Add("pHigh", "Pred: High");
            dgvMatrix.Columns.Add("total", "Row Total");

            for (int r = 0; r < 3; r++)
            {
                int total = mat[r, 0] + mat[r, 1] + mat[r, 2];
                dgvMatrix.Rows.Add($"Actual: {TierNames[r]}", mat[r, 0], mat[r, 1], mat[r, 2], total);
                // Highlight the diagonal (correct predictions) in light green
                dgvMatrix.Rows[r].Cells[r + 1].Style.BackColor = Color.LightGreen;
            }

            // Per-class metrics
            rtbMetrics.Clear();
            rtbMetrics.AppendText("Per-Class Precision, Recall & F1 (test set)\r\n");
            rtbMetrics.AppendText(new string('-', 50) + "\r\n");

            for (int c = 0; c < 3; c++)
            {
                int tp = mat[c, c];
                int fp = (mat[0, c] + mat[1, c] + mat[2, c]) - tp;
                int fn = (mat[c, 0] + mat[c, 1] + mat[c, 2]) - tp;

                double prec = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0;
                double rec  = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0;
                double f1   = (prec + rec)  > 0 ? 2 * prec * rec / (prec + rec) : 0;

                rtbMetrics.AppendText(
                    $"  [{TierNames[c],6}]  Precision={prec:P1}  Recall={rec:P1}  F1={f1:F3}\r\n");
            }

            int totalCorrect = mat[0, 0] + mat[1, 1] + mat[2, 2];
            int totalItems   = 0;
            for (int r = 0; r < 3; r++)
                for (int cc = 0; cc < 3; cc++) totalItems += mat[r, cc];
            double overallAcc = totalItems > 0 ? (double)totalCorrect / totalItems : 0;

            rtbMetrics.AppendText(new string('-', 50) + "\r\n");
            rtbMetrics.AppendText(
                $"  Overall Accuracy = {overallAcc:P2}  ({totalCorrect}/{totalItems} correct)\r\n");
        }

        // Custom Candy Prediction
        private void BtnPredict_Click(object? sender, EventArgs e)
        {
            if (_nn == null || _normalizer == null) return;

            double[] raw = new double[CSVLoader.NumFeatures];
            for (int i = 0; i < 9; i++)
                raw[i] = chkBinaryFeatures[i].Checked ? 1.0 : 0.0;
            raw[9]  = trkContinuous[0].Value / 100.0;
            raw[10] = trkContinuous[1].Value / 100.0;

            double[] norm  = _normalizer.ApplySingle(raw);
            double[] probs = _nn.ComputeOutputs(norm);
            int      tier  = ArgMax(probs);

            lblPredictedTier.Text = $"Predicted Tier: {TierNames[tier].ToUpper()} popularity";
            lblConfidence.Text    = $"Confidence: {probs[tier]:P1}";
            probBars.SetValues(probs[0], probs[1], probs[2], _lowCut, _highCut);
            probBars.Invalidate();
        }

        // Cell Formatting — just colour the Result and Predicted columns text
        private void Dgv_CellFormatting(object? sender, DataGridViewCellFormattingEventArgs e)
        {
            if (sender is not DataGridView dgv) return;
            if (e.RowIndex < 0 || e.RowIndex >= dgv.Rows.Count) return;

            bool ok = dgv.Rows[e.RowIndex].Tag is bool b && b;

            if (dgv.Columns[e.ColumnIndex].Name == "Result")
            {
                e.CellStyle.ForeColor = ok ? Color.Green : Color.Red;
                e.CellStyle.Font      = new Font(dgv.Font, FontStyle.Bold);
            }
            else if (dgv.Columns[e.ColumnIndex].Name == "Predicted")
            {
                e.CellStyle.ForeColor = ok ? Color.Green : Color.Red;
            }
        }

        // Helpers
        private void SetResultTabsEnabled(bool enabled)
        {
            tabTestPreds.Enabled  = enabled;
            tabTrainPreds.Enabled = enabled;
            tabMatrix.Enabled     = enabled;
        }

        private void Log(string msg) =>
            rtbLog.AppendText($"[{DateTime.Now:HH:mm:ss}] {msg}\n");

        private void AppendLog(string msg)
        {
            if (rtbLog.InvokeRequired)
                rtbLog.Invoke(() => Log(msg));
            else
                Log(msg);
        }

        private static int ArgMax(double[] v)
        {
            int idx = 0;
            for (int i = 1; i < v.Length; i++)
                if (v[i] > v[idx]) idx = i;
            return idx;
        }

        private static void SplitTrainTest(double[][] allData, string[] allNames,
            double trainPct, int seed,
            out double[][] trainData, out double[][] testData,
            out string[] trainNames, out string[] testNames)
        {
            var rnd    = new Random(seed);
            int n      = allData.Length;
            int nTrain = (int)(n * trainPct);

            var copyData  = (double[][])allData.Clone();
            var copyNames = (string[])allNames.Clone();

            for (int i = 0; i < n; i++)
            {
                int r = rnd.Next(i, n);
                (copyData[r],  copyData[i])  = (copyData[i],  copyData[r]);
                (copyNames[r], copyNames[i]) = (copyNames[i], copyNames[r]);
            }

            trainData  = copyData[..nTrain];
            testData   = copyData[nTrain..];
            trainNames = copyNames[..nTrain];
            testNames  = copyNames[nTrain..];
        }
    }
}
