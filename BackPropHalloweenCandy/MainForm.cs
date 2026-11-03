using System;
using System.Drawing;
using System.Windows.Forms;
using System.Threading.Tasks;

namespace BackPropHalloweenCandy
{
    public partial class MainForm : Form
    {
        // ── Model state ──────────────────────────────────────────────────────
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

        // ── Controls ─────────────────────────────────────────────────────────
        private TabControl tabControl = null!;
        private TabPage tabTrain = null!, tabTestPreds = null!,
                        tabTrainPreds = null!, tabMatrix = null!, tabPredict = null!;

        // Training tab
        private RichTextBox rtbLog = null!;
        private Button btnTrain = null!;
        private ProgressBar pbTraining = null!;
        private Label lblStatus = null!;
        private Panel pnlHyperparams = null!;
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

        // ─────────────────────────────────────────────────────────────────────
        // UI Construction
        // ─────────────────────────────────────────────────────────────────────
        private void BuildUI()
        {
            this.Text = "🎃 Halloween Candy Popularity Predictor — Backpropagation Demo";
            this.Size = new Size(1050, 700);
            this.MinimumSize = new Size(900, 600);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.Font = new Font("Segoe UI", 9f);
            this.BackColor = Color.FromArgb(30, 30, 30);
            this.ForeColor = Color.WhiteSmoke;

            tabControl = new TabControl
            {
                Dock = DockStyle.Fill,
                Appearance = TabAppearance.FlatButtons,
                ItemSize = new Size(160, 32),
                SizeMode = TabSizeMode.Fixed
            };

            tabTrain      = MakeTab("⚙  Training",           Color.FromArgb(45, 45, 48));
            tabTestPreds  = MakeTab("📋 Test Predictions",    Color.FromArgb(45, 45, 48));
            tabTrainPreds = MakeTab("📋 Train Predictions",   Color.FromArgb(45, 45, 48));
            tabMatrix     = MakeTab("📊 Confusion Matrix",    Color.FromArgb(45, 45, 48));
            tabPredict    = MakeTab("🍬 Predict Custom Candy", Color.FromArgb(45, 45, 48));

            tabControl.TabPages.AddRange(new[] { tabTrain, tabTestPreds, tabTrainPreds, tabMatrix, tabPredict });

            BuildTrainingTab();
            BuildPredictionsTab(tabTestPreds, out dgvTestPreds, out lblTestSummary);
            BuildPredictionsTab(tabTrainPreds, out dgvTrainPreds, out lblTrainSummary);
            BuildConfusionMatrixTab();
            BuildCustomCandyTab();

            this.Controls.Add(tabControl);

            // Disable result tabs until trained
            SetResultTabsEnabled(false);
        }

        private TabPage MakeTab(string text, Color backColor)
        {
            return new TabPage(text) { BackColor = backColor, ForeColor = Color.WhiteSmoke, UseVisualStyleBackColor = false };
        }

        // ── Training Tab ─────────────────────────────────────────────────────
        private void BuildTrainingTab()
        {
            var pnlOuter = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2,
                RowCount = 1,
                BackColor = Color.Transparent
            };
            pnlOuter.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 260));
            pnlOuter.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

            // ── Left: Hyperparameters ─────────────────────────────────────────
            pnlHyperparams = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.FromArgb(37, 37, 38),
                Padding = new Padding(14)
            };

            int y = 10;
            var lblTitle = new Label { Text = "Hyperparameters", Font = new Font("Segoe UI", 10f, FontStyle.Bold), ForeColor = Color.Orange, Location = new Point(10, y), AutoSize = true };
            pnlHyperparams.Controls.Add(lblTitle);
            y += 32;

            nudEpochs = AddLabeledNumeric(pnlHyperparams, "Max Epochs:", ref y, 1, 50000, 1000, 0);
            nudHidden = AddLabeledNumeric(pnlHyperparams, "Hidden Nodes:", ref y, 2, 100, 5, 0);
            nudLearnRate = AddLabeledNumeric(pnlHyperparams, "Learn Rate:", ref y, 0.001m, 1m, 0.05m, 4);
            nudMomentum = AddLabeledNumeric(pnlHyperparams, "Momentum:", ref y, 0m, 1m, 0.01m, 4);
            nudWeightDecay = AddLabeledNumeric(pnlHyperparams, "Weight Decay:", ref y, 0m, 1m, 0.01m, 4);

            chkCrossEntropy = new CheckBox
            {
                Text = "Use Cross-Entropy Loss",
                ForeColor = Color.WhiteSmoke,
                Checked = true,
                Location = new Point(10, y),
                AutoSize = true
            };
            pnlHyperparams.Controls.Add(chkCrossEntropy);
            y += 30;

            y += 10;
            btnTrain = new Button
            {
                Text = "▶  Train Network",
                Location = new Point(10, y),
                Size = new Size(220, 36),
                BackColor = Color.FromArgb(0, 122, 204),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Font = new Font("Segoe UI", 10f, FontStyle.Bold),
                Cursor = Cursors.Hand
            };
            btnTrain.FlatAppearance.BorderSize = 0;
            btnTrain.Click += BtnTrain_Click;
            pnlHyperparams.Controls.Add(btnTrain);
            y += 44;

            pbTraining = new ProgressBar
            {
                Location = new Point(10, y),
                Size = new Size(220, 16),
                Style = ProgressBarStyle.Marquee,
                MarqueeAnimationSpeed = 0,
                BackColor = Color.FromArgb(37, 37, 38)
            };
            pnlHyperparams.Controls.Add(pbTraining);
            y += 24;

            lblStatus = new Label
            {
                Text = "Ready — click Train to begin.",
                ForeColor = Color.Silver,
                Location = new Point(10, y),
                Size = new Size(230, 40),
                AutoSize = false
            };
            pnlHyperparams.Controls.Add(lblStatus);

            // ── Right: Log ────────────────────────────────────────────────────
            var pnlLog = new Panel { Dock = DockStyle.Fill, BackColor = Color.Transparent };

            var lblLog = new Label { Text = "Training Log", ForeColor = Color.Orange, Font = new Font("Segoe UI", 9f, FontStyle.Bold), Dock = DockStyle.Top, Padding = new Padding(4, 6, 0, 0), Height = 24 };

            rtbLog = new RichTextBox
            {
                Dock = DockStyle.Fill,
                BackColor = Color.FromArgb(20, 20, 20),
                ForeColor = Color.LightGreen,
                Font = new Font("Consolas", 9f),
                ReadOnly = true,
                BorderStyle = BorderStyle.None
            };

            pnlLog.Controls.Add(rtbLog);
            pnlLog.Controls.Add(lblLog);

            pnlOuter.Controls.Add(pnlHyperparams, 0, 0);
            pnlOuter.Controls.Add(pnlLog, 1, 0);
            tabTrain.Controls.Add(pnlOuter);
        }

        private NumericUpDown AddLabeledNumeric(Panel parent, string labelText, ref int y, decimal min, decimal max, decimal val, int decimals)
        {
            parent.Controls.Add(new Label { Text = labelText, ForeColor = Color.Silver, Location = new Point(10, y), AutoSize = true });
            y += 20;
            var nud = new NumericUpDown
            {
                Location = new Point(10, y),
                Size = new Size(220, 24),
                Minimum = min, Maximum = max, Value = val,
                DecimalPlaces = decimals,
                Increment = decimals == 0 ? 1 : (decimal)Math.Pow(10, -decimals),
                BackColor = Color.FromArgb(50, 50, 55),
                ForeColor = Color.WhiteSmoke,
                BorderStyle = BorderStyle.FixedSingle
            };
            parent.Controls.Add(nud);
            y += 30;
            return nud;
        }

        // ── Predictions Tab ───────────────────────────────────────────────────
        private void BuildPredictionsTab(TabPage tab, out DataGridView dgv, out Label summaryLabel)
        {
            var pnl = new TableLayoutPanel { Dock = DockStyle.Fill, RowCount = 2, ColumnCount = 1, BackColor = Color.Transparent };
            pnl.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            pnl.RowStyles.Add(new RowStyle(SizeType.Absolute, 36));

            dgv = new DataGridView
            {
                Dock = DockStyle.Fill,
                ReadOnly = true,
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false,
                AllowUserToResizeRows = false,
                ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.DisableResizing,
                ColumnHeadersHeight = 32,
                RowHeadersVisible = false,
                SelectionMode = DataGridViewSelectionMode.FullRowSelect,
                BackgroundColor = Color.FromArgb(30, 30, 30),
                GridColor = Color.FromArgb(60, 60, 60),
                BorderStyle = BorderStyle.None,
                Font = new Font("Consolas", 8.5f),
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill
            };

            dgv.ColumnHeadersDefaultCellStyle = new DataGridViewCellStyle
            {
                BackColor = Color.FromArgb(0, 100, 160),
                ForeColor = Color.White,
                Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                Alignment = DataGridViewContentAlignment.MiddleCenter
            };
            dgv.DefaultCellStyle = new DataGridViewCellStyle
            {
                BackColor = Color.FromArgb(37, 37, 38),
                ForeColor = Color.WhiteSmoke,
                SelectionBackColor = Color.FromArgb(0, 122, 204),
                SelectionForeColor = Color.White,
                Alignment = DataGridViewContentAlignment.MiddleCenter
            };
            dgv.AlternatingRowsDefaultCellStyle = new DataGridViewCellStyle
            {
                BackColor = Color.FromArgb(45, 45, 48)
            };

            var captionStyle = new DataGridViewCellStyle { Alignment = DataGridViewContentAlignment.MiddleLeft };
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "Name",      HeaderText = "Candy Name",  FillWeight = 25, DefaultCellStyle = captionStyle });
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "Actual",    HeaderText = "Actual Tier", FillWeight = 10 });
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "Predicted", HeaderText = "Predicted",   FillWeight = 10 });
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "ProbLow",   HeaderText = "P(Low)",      FillWeight = 10 });
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "ProbMed",   HeaderText = "P(Medium)",   FillWeight = 10 });
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "ProbHigh",  HeaderText = "P(High)",     FillWeight = 10 });
            dgv.Columns.Add(new DataGridViewTextBoxColumn { Name = "Result",    HeaderText = "Result",      FillWeight = 7 });
            dgv.CellFormatting += Dgv_CellFormatting;

            summaryLabel = new Label
            {
                Dock = DockStyle.Fill,
                ForeColor = Color.Silver,
                TextAlign = ContentAlignment.MiddleLeft,
                Padding = new Padding(8, 0, 0, 0),
                Font = new Font("Segoe UI", 9f)
            };

            pnl.Controls.Add(dgv, 0, 0);
            pnl.Controls.Add(summaryLabel, 0, 1);
            tab.Controls.Add(pnl);
        }

        // ── Confusion Matrix Tab ──────────────────────────────────────────────
        private void BuildConfusionMatrixTab()
        {
            var split = new SplitContainer
            {
                Dock = DockStyle.Fill,
                Orientation = Orientation.Horizontal,
                BackColor = Color.Transparent,
                BorderStyle = BorderStyle.None,
                SplitterWidth = 6
            };

            dgvMatrix = new DataGridView
            {
                Dock = DockStyle.Fill,
                ReadOnly = true,
                AllowUserToAddRows = false,
                AllowUserToResizeRows = false,
                RowHeadersVisible = false,
                BackgroundColor = Color.FromArgb(30, 30, 30),
                GridColor = Color.FromArgb(60, 60, 60),
                BorderStyle = BorderStyle.None,
                Font = new Font("Consolas", 9f),
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.AllCells,
                ColumnHeadersHeight = 32
            };
            dgvMatrix.ColumnHeadersDefaultCellStyle = new DataGridViewCellStyle
            {
                BackColor = Color.FromArgb(0, 100, 160),
                ForeColor = Color.White,
                Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                Alignment = DataGridViewContentAlignment.MiddleCenter
            };
            dgvMatrix.DefaultCellStyle = new DataGridViewCellStyle
            {
                BackColor = Color.FromArgb(37, 37, 38),
                ForeColor = Color.WhiteSmoke,
                Alignment = DataGridViewContentAlignment.MiddleCenter,
                Font = new Font("Consolas", 9f)
            };

            rtbMetrics = new RichTextBox
            {
                Dock = DockStyle.Fill,
                ReadOnly = true,
                BackColor = Color.FromArgb(20, 20, 20),
                ForeColor = Color.WhiteSmoke,
                Font = new Font("Consolas", 9.5f),
                BorderStyle = BorderStyle.None
            };

            split.Panel1.Controls.Add(dgvMatrix);
            split.Panel2.Controls.Add(rtbMetrics);
            split.SplitterDistance = 220;

            tabMatrix.Controls.Add(split);
        }

        // ── Custom Candy Tab ──────────────────────────────────────────────────
        private void BuildCustomCandyTab()
        {
            chkBinaryFeatures = new CheckBox[9];
            trkContinuous     = new TrackBar[2];
            lblContinuousVal  = new Label[2];

            var scrollPanel = new Panel { Dock = DockStyle.Left, Width = 380, AutoScroll = true, BackColor = Color.FromArgb(37, 37, 38) };
            var pnlRight    = new Panel { Dock = DockStyle.Fill, BackColor = Color.FromArgb(30, 30, 30), Padding = new Padding(16) };

            int y = 10;
            var lbl = new Label { Text = "Define Your Candy Recipe", Font = new Font("Segoe UI", 10f, FontStyle.Bold), ForeColor = Color.Orange, Location = new Point(12, y), AutoSize = true };
            scrollPanel.Controls.Add(lbl);
            y += 30;

            var sep = new Label { Text = "── Binary Attributes ─────────────────", ForeColor = Color.DimGray, Location = new Point(12, y), AutoSize = true, Font = new Font("Segoe UI", 8.5f) };
            scrollPanel.Controls.Add(sep);
            y += 22;

            for (int i = 0; i < 9; i++)
            {
                chkBinaryFeatures[i] = new CheckBox
                {
                    Text = FeatureNames[i],
                    ForeColor = Color.WhiteSmoke,
                    Location = new Point(16, y),
                    AutoSize = true
                };
                scrollPanel.Controls.Add(chkBinaryFeatures[i]);
                y += 26;
            }

            y += 8;
            var sep2 = new Label { Text = "── Continuous Attributes ─────────────", ForeColor = Color.DimGray, Location = new Point(12, y), AutoSize = true, Font = new Font("Segoe UI", 8.5f) };
            scrollPanel.Controls.Add(sep2);
            y += 22;

            string[] contLabels = { "Sugar Percentile (0.0 – 1.0)", "Price Percentile (0.0 – 1.0)" };
            for (int i = 0; i < 2; i++)
            {
                int capturedI = i;
                scrollPanel.Controls.Add(new Label { Text = contLabels[i], ForeColor = Color.Silver, Location = new Point(12, y), AutoSize = true });
                y += 20;

                lblContinuousVal[i] = new Label { Text = "0.50", ForeColor = Color.Orange, Location = new Point(310, y - 2), AutoSize = true };
                scrollPanel.Controls.Add(lblContinuousVal[i]);

                trkContinuous[i] = new TrackBar
                {
                    Location = new Point(12, y),
                    Size = new Size(290, 36),
                    Minimum = 0, Maximum = 100, Value = 50,
                    TickFrequency = 10,
                    SmallChange = 1,
                    LargeChange = 10
                };
                trkContinuous[i].ValueChanged += (s, e) =>
                {
                    lblContinuousVal[capturedI].Text = (trkContinuous[capturedI].Value / 100.0).ToString("F2");
                };
                scrollPanel.Controls.Add(trkContinuous[i]);
                y += 44;
            }

            y += 8;
            btnPredict = new Button
            {
                Text = "🔮  Predict Popularity Tier",
                Location = new Point(12, y),
                Size = new Size(340, 38),
                BackColor = Color.FromArgb(176, 86, 0),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Font = new Font("Segoe UI", 10f, FontStyle.Bold),
                Cursor = Cursors.Hand,
                Enabled = false
            };
            btnPredict.FlatAppearance.BorderSize = 0;
            btnPredict.Click += BtnPredict_Click;
            scrollPanel.Controls.Add(btnPredict);

            // Right-side result panel
            lblPredictedTier = new Label
            {
                Text = "Train the network first, then predict!",
                ForeColor = Color.Silver,
                Font = new Font("Segoe UI", 14f, FontStyle.Bold),
                TextAlign = ContentAlignment.TopCenter,
                Dock = DockStyle.Top,
                Height = 50,
                Padding = new Padding(0, 8, 0, 0)
            };

            lblConfidence = new Label
            {
                Text = "",
                ForeColor = Color.WhiteSmoke,
                Font = new Font("Segoe UI", 10f),
                TextAlign = ContentAlignment.TopCenter,
                Dock = DockStyle.Top,
                Height = 28
            };

            probBars = new ProbabilityBarPanel
            {
                Dock = DockStyle.Top,
                Height = 110,
                BackColor = Color.Transparent
            };

            pnlRight.Controls.Add(probBars);
            pnlRight.Controls.Add(lblConfidence);
            pnlRight.Controls.Add(lblPredictedTier);

            tabPredict.Controls.Add(pnlRight);
            tabPredict.Controls.Add(scrollPanel);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Training Logic
        // ─────────────────────────────────────────────────────────────────────
        private async void BtnTrain_Click(object? sender, EventArgs e)
        {
            btnTrain.Enabled = false;
            pbTraining.MarqueeAnimationSpeed = 30;
            SetResultTabsEnabled(false);
            rtbLog.Clear();

            int    maxEpochs    = (int)nudEpochs.Value;
            int    numHidden    = (int)nudHidden.Value;
            double learnRate    = (double)nudLearnRate.Value;
            double momentum     = (double)nudMomentum.Value;
            double weightDecay  = (double)nudWeightDecay.Value;
            bool   crossEntropy = chkCrossEntropy.Checked;

            Log($"Loading candy-data.csv …");

            try
            {
                await Task.Run(() =>
                {
                    int numInput  = CSVLoader.NumFeatures;
                    int numOutput = CSVLoader.NumClasses;

                    CSVLoader.Load("candy-data.csv", out double[][] features, out double[] winPercents, out string[] names);

                    double[][] allData = CSVLoader.BuildTierRows(features, winPercents, out double lowCut, out double highCut);
                    _lowCut  = lowCut;
                    _highCut = highCut;

                    AppendLog($"Loaded {allData.Length} candies | {numInput} features");
                    AppendLog($"Tiers: Low < {lowCut:F2}% | Medium [{lowCut:F2}% – {highCut:F2}%) | High ≥ {highCut:F2}%");

                    SplitTrainTest(allData, names, 0.80, 1,
                        out double[][] trainData, out double[][] testData,
                        out string[] trainNames, out string[] testNames);

                    AppendLog($"Split: {trainData.Length} train / {testData.Length} test");

                    var normalizer = new Normalizer(numInput);
                    normalizer.Fit(trainData);
                    normalizer.Apply(trainData);
                    normalizer.Apply(testData);
                    AppendLog("Features normalised [0,1] on train data.");

                    var nn = new NeuralNetwork(numInput, numHidden, numOutput);
                    AppendLog($"Network: {numInput}-{numHidden}-{numOutput}  |  epochs={maxEpochs}  lr={learnRate:F4}  mom={momentum:F4}  wd={weightDecay:F4}  ce={crossEntropy}");
                    AppendLog("Training …");

                    nn.Train(trainData, testData, maxEpochs, learnRate, momentum,
                             crossEntropy, weightDecay, 100, "training_log.csv");

                    double trainAcc = nn.Accuracy(trainData);
                    double testAcc  = nn.Accuracy(testData);
                    AppendLog($"Done!  Train acc = {trainAcc:P2}  |  Test acc = {testAcc:P2}");

                    _nn         = nn;
                    _normalizer = normalizer;
                    _trainData  = trainData;
                    _testData   = testData;
                    _trainNames = trainNames;
                    _testNames  = testNames;
                });

                PopulatePredictionsGrid(dgvTestPreds,  lblTestSummary,  _testData!,  _testNames!);
                PopulatePredictionsGrid(dgvTrainPreds, lblTrainSummary, _trainData!, _trainNames!);
                PopulateConfusionMatrix();
                SetResultTabsEnabled(true);
                btnPredict.Enabled = true;
                lblPredictedTier.Text = "Enter recipe on the left and click Predict!";
                lblStatus.Text = "Training complete ✓";
                lblStatus.ForeColor = Color.LightGreen;
            }
            catch (Exception ex)
            {
                AppendLog($"ERROR: {ex.Message}");
                lblStatus.Text = "Error — see log.";
                lblStatus.ForeColor = Color.OrangeRed;
            }

            btnTrain.Enabled = true;
            pbTraining.MarqueeAnimationSpeed = 0;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Populate Results
        // ─────────────────────────────────────────────────────────────────────
        private void PopulatePredictionsGrid(DataGridView dgv, Label summary, double[][] data, string[] names)
        {
            dgv.Rows.Clear();
            int correct = 0;
            int numFeatures = CSVLoader.NumFeatures;
            int numClasses  = CSVLoader.NumClasses;

            for (int i = 0; i < data.Length; i++)
            {
                double[] x = new double[numFeatures];
                Array.Copy(data[i], x, numFeatures);

                double[] probs = _nn!.ComputeOutputs(x);
                int pred   = ArgMax(probs);
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
                    ok ? "✓ OK" : "✗ MISS");

                dgv.Rows[i].Tag = ok; // used for colouring
            }

            double acc = (double)correct / data.Length;
            summary.Text = $"  {correct} / {data.Length} correct — Accuracy: {acc:P2}";
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
                int pred = ArgMax(probs);
                int actual = 0;
                for (int c = 0; c < 3; c++)
                    if (row[numFeatures + c] == 1.0) { actual = c; break; }
                mat[actual, pred]++;
            }

            dgvMatrix.Rows.Clear();
            dgvMatrix.Columns.Clear();
            dgvMatrix.Columns.Add("row",    "");
            dgvMatrix.Columns.Add("pLow",   "Pred: Low");
            dgvMatrix.Columns.Add("pMed",   "Pred: Medium");
            dgvMatrix.Columns.Add("pHigh",  "Pred: High");
            dgvMatrix.Columns.Add("total",  "Row Total");

            for (int r = 0; r < 3; r++)
            {
                int total = mat[r, 0] + mat[r, 1] + mat[r, 2];
                dgvMatrix.Rows.Add($"Actual: {TierNames[r]}", mat[r, 0], mat[r, 1], mat[r, 2], total);
                // Highlight diagonal (correct predictions)
                dgvMatrix.Rows[r].Cells[r + 1].Style.BackColor = Color.FromArgb(0, 100, 50);
                dgvMatrix.Rows[r].Cells[r + 1].Style.ForeColor = Color.White;
            }

            // Per-class metrics text
            rtbMetrics.Clear();
            rtbMetrics.SelectionFont  = new Font("Consolas", 9.5f, FontStyle.Bold);
            rtbMetrics.SelectionColor = Color.Orange;
            rtbMetrics.AppendText("Per-Class Precision, Recall & F1 (on test set)\r\n");
            rtbMetrics.AppendText(new string('─', 55) + "\r\n");

            for (int c = 0; c < 3; c++)
            {
                int tp = mat[c, c];
                int fp = (mat[0, c] + mat[1, c] + mat[2, c]) - tp;
                int fn = (mat[c, 0] + mat[c, 1] + mat[c, 2]) - tp;

                double prec = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0;
                double rec  = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0;
                double f1   = (prec + rec)  > 0 ? 2 * prec * rec / (prec + rec) : 0;

                rtbMetrics.SelectionFont  = new Font("Consolas", 9.5f, FontStyle.Bold);
                rtbMetrics.SelectionColor = Color.SkyBlue;
                rtbMetrics.AppendText($"  Tier [{TierNames[c],6}]  ");

                rtbMetrics.SelectionFont  = new Font("Consolas", 9.5f);
                rtbMetrics.SelectionColor = Color.WhiteSmoke;
                rtbMetrics.AppendText($"Precision = {prec:P1}   Recall = {rec:P1}   F1 = {f1:F3}\r\n");
            }

            // Overall accuracy
            int totalCorrect = mat[0, 0] + mat[1, 1] + mat[2, 2];
            int totalItems   = 0;
            for (int r = 0; r < 3; r++) for (int cc = 0; cc < 3; cc++) totalItems += mat[r, cc];
            double overallAcc = totalItems > 0 ? (double)totalCorrect / totalItems : 0;

            rtbMetrics.AppendText(new string('─', 55) + "\r\n");
            rtbMetrics.SelectionFont  = new Font("Consolas", 9.5f, FontStyle.Bold);
            rtbMetrics.SelectionColor = Color.LightGreen;
            rtbMetrics.AppendText($"  Overall Accuracy = {overallAcc:P2}  ({totalCorrect}/{totalItems} correct)\r\n");
        }

        // ─────────────────────────────────────────────────────────────────────
        // Custom Candy Prediction
        // ─────────────────────────────────────────────────────────────────────
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

            Color tierColor = tier == 2 ? Color.LimeGreen : tier == 1 ? Color.Gold : Color.OrangeRed;

            lblPredictedTier.Text      = $"Predicted Tier: {TierNames[tier].ToUpper()} POPULARITY";
            lblPredictedTier.ForeColor = tierColor;
            lblConfidence.Text         = $"Confidence: {probs[tier]:P1}";
            probBars.SetValues(probs[0], probs[1], probs[2], _lowCut, _highCut);
            probBars.Invalidate();
        }

        // ─────────────────────────────────────────────────────────────────────
        // Cell Formatting
        // ─────────────────────────────────────────────────────────────────────
        private void Dgv_CellFormatting(object? sender, DataGridViewCellFormattingEventArgs e)
        {
            if (sender is not DataGridView dgv) return;
            if (e.RowIndex < 0 || e.RowIndex >= dgv.Rows.Count) return;

            var row = dgv.Rows[e.RowIndex];
            bool ok = row.Tag is bool b && b;

            if (dgv.Columns[e.ColumnIndex].Name == "Result")
            {
                e.CellStyle.ForeColor = ok ? Color.LimeGreen : Color.OrangeRed;
                e.CellStyle.Font      = new Font("Segoe UI", 9f, FontStyle.Bold);
            }
            else if (dgv.Columns[e.ColumnIndex].Name == "Predicted")
            {
                e.CellStyle.ForeColor = ok ? Color.LightGreen : Color.OrangeRed;
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helpers
        // ─────────────────────────────────────────────────────────────────────
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

        private static void SplitTrainTest(double[][] allData, string[] allNames, double trainPct, int seed,
            out double[][] trainData, out double[][] testData,
            out string[] trainNames, out string[] testNames)
        {
            var rnd = new Random(seed);
            int n       = allData.Length;
            int nTrain  = (int)(n * trainPct);
            int nTest   = n - nTrain;

            var copyData  = (double[][])allData.Clone();
            var copyNames = (string[])allNames.Clone();

            for (int i = 0; i < n; i++)
            {
                int r = rnd.Next(i, n);
                (copyData[r], copyData[i])   = (copyData[i], copyData[r]);
                (copyNames[r], copyNames[i]) = (copyNames[i], copyNames[r]);
            }

            trainData  = copyData[..nTrain];
            testData   = copyData[nTrain..];
            trainNames = copyNames[..nTrain];
            testNames  = copyNames[nTrain..];
        }
    }
}
