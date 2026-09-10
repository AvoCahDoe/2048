import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export type AgentId = 'random' | 'heuristic' | 'expectimax' | 'dqn';

export interface SimState {
  board: number[][];
  score: number;
  max_tile?: number;
  done?: boolean;
  won?: boolean;
  legal_actions?: number[];
  seed?: number;
  action?: number | null;
  action_name?: string | null;
  gained?: number;
}

export interface RunResult {
  agent: string;
  summary: Record<string, unknown>;
  episodes: Array<{
    episode: number;
    score: number;
    max_tile: number;
    steps: number;
    won: boolean;
  }>;
}

@Injectable({ providedIn: 'root' })
export class LabApi {
  private http = inject(HttpClient);
  private base = environment.apiBase;

  health(): Observable<{ status: string }> {
    return this.http.get<{ status: string }>(`${this.base}/health`);
  }

  agents(): Observable<{ agents: Array<{ id: string; label: string; blurb: string; ready?: boolean }> }> {
    return this.http.get<{ agents: Array<{ id: string; label: string; blurb: string; ready?: boolean }> }>(
      `${this.base}/agents`
    );
  }

  reset(seed?: number): Observable<SimState> {
    return this.http.post<SimState>(`${this.base}/sim/reset`, { seed: seed ?? null });
  }

  step(body: {
    board: number[][];
    score: number;
    agent: AgentId;
    depth?: number;
    seed?: number;
  }): Observable<SimState> {
    return this.http.post<SimState>(`${this.base}/sim/step`, body);
  }

  run(body: {
    agent: AgentId;
    episodes: number;
    seed: number;
    depth?: number;
  }): Observable<RunResult> {
    return this.http.post<RunResult>(`${this.base}/sim/run`, body);
  }

  bakeoff(): Observable<{
    summary: Record<string, Record<string, unknown>>;
    episodes: Array<Record<string, unknown>>;
  }> {
    return this.http.get<{
      summary: Record<string, Record<string, unknown>>;
      episodes: Array<Record<string, unknown>>;
    }>(`${this.base}/results/bakeoff`);
  }

  dqn(run = 'dqn_default'): Observable<{
    run: string;
    summary: Record<string, unknown>;
    metrics: Array<Record<string, unknown>>;
  }> {
    return this.http.get<{
      run: string;
      summary: Record<string, unknown>;
      metrics: Array<Record<string, unknown>>;
    }>(`${this.base}/results/dqn`, { params: { run } });
  }

  ablations(): Observable<{ runs: Array<Record<string, unknown>> }> {
    return this.http.get<{ runs: Array<Record<string, unknown>> }>(
      `${this.base}/results/ablations`
    );
  }
}
