using System;
using System.Drawing;
using System.Windows.Forms;

namespace BackPropHalloweenCandy
{
    /// <summary>
    /// Custom owner-drawn panel that renders three horizontal probability bars
    /// for Low / Medium / High tiers. Uses only System.Drawing.
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
            _pLow    = pLow;
            _pMed    = pMed;
            _pHigh   = pHigh;
            _lowCut  = lowCut;
            _highCut = highCut;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            var g = e.Graphics;

            if (_pLow == 0 && _pMed == 0 && _pHigh == 0) return;

            double[] probs  = { _pLow, _pMed, _pHigh };
            string[] labels =
            {
                $"Low    (< {_lowCut:F1}%)",
                $"Medium ({_lowCut:F1}% - {_highCut:F1}%)",
                $"High   (>= {_highCut:F1}%)"
            };

            int barHeight = 22;
            int gap       = 8;
            int labelW    = 200;
            int barMaxW   = this.Width - labelW - 70;
            int startY    = 6;

            for (int i = 0; i < 3; i++)
            {
                int y = startY + i * (barHeight + gap);

                // Label
                g.DrawString(labels[i], this.Font, SystemBrushes.ControlText, 2, y + 3);

                // Track background (using a standard control-shadow color)
                var trackRect = new Rectangle(labelW, y, barMaxW, barHeight);
                g.FillRectangle(SystemBrushes.ControlDark, trackRect);
                g.DrawRectangle(SystemPens.ControlDarkDark, trackRect);

                // Filled portion
                int filledW = (int)(probs[i] * barMaxW);
                if (filledW > 0)
                {
                    g.FillRectangle(SystemBrushes.ActiveCaption,
                        labelW, y, filledW, barHeight);
                }

                // Percentage text
                string pct = probs[i].ToString("P1");
                g.DrawString(pct, this.Font, SystemBrushes.ControlText,
                    labelW + barMaxW + 4, y + 3);
            }
        }
    }
}
