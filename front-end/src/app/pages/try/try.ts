import { Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AgentId, LabApi, RunResult, SimState } from '../../services/lab-api';

@Component({
  selector: 'app-try',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './try.html',
  styleUrl: './try.scss',
})
export class TryPage implements OnInit, OnDestroy {
  private api = inject(LabApi);

  agent: AgentId = 'heuristic';
  depth = 1;
  episodes = 8;
  seed = 42;
  mode: 'watch' | 'batch' = 'watch';

  board = signal<number[][]>([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ]);
  score = signal(0);
  maxTile = signal(0);
  lastAction = signal<string | null>(null);
  done = signal(false);
  busy = signal(false);
  error = signal<string | null>(null);
  apiOk = signal(false);

  batchResult = signal<RunResult | null>(null);
  watching = signal(false);

  private watchTimer: ReturnType<typeof setInterval> | null = null;

  agents: { id: AgentId; label: string }[] = [
    { id: 'random', label: 'Random' },
    { id: 'heuristic', label: 'Heuristic' },
    { id: 'expectimax', label: 'Expectimax' },
    { id: 'dqn', label: 'Double DQN' },
  ];

  ngOnInit(): void {
    this.api.health().subscribe({
      next: () => {
        this.apiOk.set(true);
        this.resetBoard();
      },
      error: () => {
        this.apiOk.set(false);
        this.error.set('API offline — start with: uv run twenty48-api');
      },
    });
  }

  ngOnDestroy(): void {
    this.stopWatch();
  }

  resetBoard(): void {
    this.stopWatch();
    this.error.set(null);
    this.busy.set(true);
    this.api.reset(this.seed).subscribe({
      next: (s) => {
        this.applyState(s);
        this.busy.set(false);
      },
      error: (e) => {
        this.busy.set(false);
        this.error.set(e?.error?.detail || 'Reset failed');
      },
    });
  }

  stepOnce(): void {
    if (this.done() || this.busy()) return;
    this.busy.set(true);
    this.api
      .step({
        board: this.board(),
        score: this.score(),
        agent: this.agent,
        depth: this.depth,
        seed: Math.floor(Math.random() * 1e9),
      })
      .subscribe({
        next: (s) => {
          this.applyState(s);
          this.busy.set(false);
          if (s.action_name) this.lastAction.set(s.action_name);
        },
        error: (e) => {
          this.busy.set(false);
          this.error.set(e?.error?.detail || 'Step failed (is DQN checkpoint present?)');
          this.stopWatch();
        },
      });
  }

  toggleWatch(): void {
    if (this.watching()) {
      this.stopWatch();
      return;
    }
    if (this.done()) this.resetBoard();
    this.watching.set(true);
    this.watchTimer = setInterval(() => {
      if (this.done()) {
        this.stopWatch();
        return;
      }
      if (!this.busy()) this.stepOnce();
    }, this.agent === 'expectimax' ? 450 : 220);
  }

  stopWatch(): void {
    this.watching.set(false);
    if (this.watchTimer) {
      clearInterval(this.watchTimer);
      this.watchTimer = null;
    }
  }

  runBatch(): void {
    this.stopWatch();
    this.error.set(null);
    this.busy.set(true);
    this.batchResult.set(null);
    this.api
      .run({
        agent: this.agent,
        episodes: this.episodes,
        seed: this.seed,
        depth: this.depth,
      })
      .subscribe({
        next: (r) => {
          this.batchResult.set(r);
          this.busy.set(false);
        },
        error: (e) => {
          this.busy.set(false);
          this.error.set(e?.error?.detail || 'Batch run failed');
        },
      });
  }

  tileClass(v: number): string {
    if (v === 0) return 'empty';
    if (v <= 4) return 't2';
    if (v <= 16) return 't8';
    if (v <= 64) return 't32';
    if (v <= 256) return 't128';
    return 't512';
  }

  summaryNum(key: string): string {
    const s = this.batchResult()?.summary;
    if (!s || s[key] == null) return '—';
    const v = s[key] as number;
    return typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(1)) : String(v);
  }

  private applyState(s: SimState): void {
    this.board.set(s.board);
    this.score.set(s.score);
    this.maxTile.set(s.max_tile ?? Math.max(...s.board.flat()));
    this.done.set(!!s.done);
  }
}
