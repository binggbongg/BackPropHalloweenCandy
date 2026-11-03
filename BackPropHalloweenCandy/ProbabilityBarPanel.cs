using System;
using System.Drawing;
using System.Windows.Forms;

namespace BackPropHalloweenCandy
{
    /// <summary>
    /// Custom owner-drawn panel that renders three horizontal probability bars
    /// for Low / Medium / High tiers with labels and percentage text.
    /// Uses only System.Drawing — no external libraries.
    /// </summary>
    public class ProbabilityBarPanel : Panel
    {
        private double _pLow, _pMed, _pHigh;
        private double _lowCut, _highCut;

        public ProbabilityBarPanel()
        {
            this.DoubleBuffered = true;
        }

        public void SetValues(double pLow, double pMed, double pHigh, double lowCut, double highCut)
        {
            _pLow   = pLow;
            _pMed   = pMed;
            _pHigh  = pHigh;
            _lowCut  = lowCut;
            _highCut = highCut;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            var g = e.Graphics;
            g.Clear(Color.FromArgb(30, 30, 30));

            if (_pLow == 0 && _pMed == 0 && _pHigh == 0) return;

            double[] probs  = { _pLow, _pMed, _pHigh };
            string[] labels = { $"Low    (< {_lowCut:F1}%)", $"Medium ({_lowCut:F1}% – {_highCut:F1}%)", $"High   (≥ {_highCut:F1}%)" };
            Color[]  colors =
            {
                Color.FromArgb(220, 80,  50),
                Color.FromArgb(220, 180, 50),
                Color.FromArgb(60,  180, 60)
            };

            int barHeight = 24;
            int gap       = 10;
            int labelW    = 220;
            int barMaxW   = this.Width - labelW - 90;
            int startY    = 8;

            using var fontLabel  = new Font("Segoe UI", 9f);
            using var fontPct    = new Font("Segoe UI", 9f, FontStyle.Bold);
            using var brushText  = new SolidBrush(Color.Silver);
            using var brushTrack = new SolidBrush(Color.FromArgb(55, 55, 58));

            for (int i = 0; i < 3; i++)
            {
                int y = startY + i * (barHeight + gap);

                // Label
                g.DrawString(labels[i], fontLabel, brushText, 4, y + 4);

                // Track background
                var trackRect = new Rectangle(labelW, y, barMaxW, barHeight);
                g.FillRectangle(brushTrack, trackRect);

                // Filled bar
                int filledW = (int)(probs[i] * barMaxW);
                if (filledW > 0)
                {
                    using var brushBar = new SolidBrush(colors[i]);
                    g.FillRectangle(brushBar, labelW, y, filledW, barHeight);
                }

                // Percentage text
                string pct = probs[i].ToString("P1");
                g.DrawString(pct, fontPct, Brushes.White, labelW + barMaxW + 6, y + 4);
            }
        }
    }
}
