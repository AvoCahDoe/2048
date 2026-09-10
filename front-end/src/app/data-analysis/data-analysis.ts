import { Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Move } from '../moves/moves';

@Component({
  selector: 'app-data-analysis',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './data-analysis.html',
  styleUrl: './data-analysis.scss',
})
export class DataAnalysis implements OnChanges {
  @Input() moves: Move[] = [];
  @Input() score = 0;

  scorePath = '';
  scorePoints: { x: number; y: number }[] = [];
  directions: { label: string; count: number; pct: number; color: string }[] = [];

  private readonly colors: Record<string, string> = {
    Left: '#c084fc',
    Right: '#a855f7',
    Up: '#7c3aed',
    Down: '#5b21b6',
  };

  ngOnChanges(): void {
    this.rebuildScoreSeries();
    this.rebuildDirections();
  }

  private rebuildScoreSeries(): void {
    const w = 260;
    const h = 100;
    if (!this.moves.length) {
      this.scorePath = '';
      this.scorePoints = [];
      return;
    }
    let running = 0;
    const series = this.moves.map((m, i) => {
      running += m.scoreGained;
      return { i, s: running };
    });
    const maxS = Math.max(...series.map((p) => p.s), 1);
    const n = series.length;
    this.scorePoints = series.map((p) => ({
      x: n === 1 ? w / 2 : (p.i / (n - 1)) * w,
      y: h - (p.s / maxS) * (h - 8) - 4,
    }));
    this.scorePath = this.scorePoints
      .map((p, idx) => `${idx === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
      .join(' ');
  }

  private rebuildDirections(): void {
    const counts: Record<string, number> = { Left: 0, Right: 0, Up: 0, Down: 0 };
    for (const m of this.moves) {
      if (m.direction in counts) counts[m.direction]++;
    }
    const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
    this.directions = Object.entries(counts).map(([label, count]) => ({
      label,
      count,
      pct: (count / total) * 100,
      color: this.colors[label],
    }));
  }
}
