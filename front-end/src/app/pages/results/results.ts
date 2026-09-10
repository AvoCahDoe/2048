import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LabApi } from '../../services/lab-api';

@Component({
  selector: 'app-results',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './results.html',
  styleUrl: './results.scss',
})
export class ResultsPage implements OnInit {
  private api = inject(LabApi);

  error = signal<string | null>(null);
  loading = signal(true);

  bakeoffRows = signal<
    Array<{ agent: string; mean: number; maxTile: number; best: number; n: number }>
  >([]);
  scoreBars = signal<Array<{ agent: string; score: number; pct: number }>>([]);
  dqnPoints = signal<Array<{ ep: number; score: number; ma: number; y: number }>>([]);
  dqnSummary = signal<Record<string, unknown> | null>(null);
  ablations = signal<Array<{ run: string; mean: number; maxTile: number; pct: number }>>([]);

  ngOnInit(): void {
    this.api.bakeoff().subscribe({
      next: (data) => {
        const rows = Object.entries(data.summary).map(([agent, s]) => ({
          agent,
          mean: Number(s['mean_score'] ?? 0),
          maxTile: Number(s['mean_max_tile'] ?? 0),
          best: Number(s['max_max_tile'] ?? 0),
          n: Number(s['n_episodes'] ?? 0),
        }));
        rows.sort((a, b) => b.mean - a.mean);
        this.bakeoffRows.set(rows);
        const max = Math.max(...rows.map((r) => r.mean), 1);
        this.scoreBars.set(rows.map((r) => ({ agent: r.agent, score: r.mean, pct: (r.mean / max) * 100 })));
        this.loadDqn();
      },
      error: () => {
        this.error.set('Could not load bakeoff — is the API running?');
        this.loading.set(false);
      },
    });
  }

  private loadDqn(): void {
    this.api.dqn('dqn_default').subscribe({
      next: (data) => {
        this.dqnSummary.set(data.summary);
        const scores = data.metrics.map((m) => Number(m['score'] ?? 0));
        const eps = data.metrics.map((m) => Number(m['episode'] ?? 0));
        const ma: number[] = [];
        let sum = 0;
        const w = 30;
        for (let i = 0; i < scores.length; i++) {
          sum += scores[i];
          if (i >= w) sum -= scores[i - w];
          ma.push(sum / Math.min(i + 1, w));
        }
        const maxS = Math.max(...scores, ...ma, 1);
        this.dqnPoints.set(
          eps.map((ep, i) => ({
            ep,
            score: scores[i],
            ma: ma[i],
            y: 100 - (ma[i] / maxS) * 90,
          }))
        );
        this.loadAblations();
      },
      error: () => {
        this.loadAblations();
      },
    });
  }

  private loadAblations(): void {
    this.api.ablations().subscribe({
      next: (data) => {
        const runs = data.runs.map((r) => ({
          run: String(r['run']),
          mean: Number(r['mean_score'] ?? 0),
          maxTile: Number(r['mean_max_tile'] ?? 0),
        }));
        const max = Math.max(...runs.map((r) => r.mean), 1);
        this.ablations.set(runs.map((r) => ({ ...r, pct: (r.mean / max) * 100 })));
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }

  sparkPath(): string {
    const pts = this.dqnPoints();
    if (pts.length < 2) return '';
    const w = 560;
    return pts
      .map((p, i) => {
        const x = (i / (pts.length - 1)) * w;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${p.y.toFixed(1)}`;
      })
      .join(' ');
  }

  fmt(v: unknown): string {
    if (typeof v === 'number') return v.toFixed(1);
    return v == null ? '—' : String(v);
  }
}
