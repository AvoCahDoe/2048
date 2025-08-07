// controls.ts
import { Component, Output, EventEmitter, OnInit, OnDestroy, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-controls',
  standalone: true,
  imports: [CommonModule],
  templateUrl:'./controls.html',
styleUrl:'./controls.scss'
}
)
export class ControlsComponent implements OnInit, OnDestroy {
  @Output() onStart = new EventEmitter<void>();
  @Output() onPause = new EventEmitter<void>();
  @Output() onReset = new EventEmitter<void>();

  @Input() canStart = true;

  gameState: 'ready' | 'playing' | 'paused' = 'ready';
  elapsedTime = 0;
  bestTime = 0;
  gamesPlayed = 0;

  private timer: any;
  private startTime = 0;

  ngOnInit() {
    this.loadStats();
  }

  ngOnDestroy() {
    this.clearTimer();
  }

  toggleGame() {
    switch (this.gameState) {
      case 'ready':
        this.startGame();
        break;
      case 'playing':
        this.pauseGame();
        break;
      case 'paused':
        this.resumeGame();
        break;
    }
  }

  startGame() {
    this.gameState = 'playing';
    this.startTime = Date.now() - this.elapsedTime;
    this.startTimer();
    this.onStart.emit();
  }

  pauseGame() {
    this.gameState = 'paused';
    this.clearTimer();
    this.onPause.emit();
  }

  resumeGame() {
    this.gameState = 'playing';
    this.startTime = Date.now() - this.elapsedTime;
    this.startTimer();
    this.onStart.emit();
  }

  resetGame() {
    this.gameState = 'ready';
    this.clearTimer();

    // Save best time if this was a completed game
    if (this.elapsedTime > 0) {
      this.gamesPlayed++;
      if (this.bestTime === 0 || this.elapsedTime < this.bestTime) {
        this.bestTime = this.elapsedTime;
      }
      this.saveStats();
    }

    this.elapsedTime = 0;
    this.onReset.emit();
  }

  private startTimer() {
    this.timer = setInterval(() => {
      this.elapsedTime = Date.now() - this.startTime;
    }, 10);
  }

  private clearTimer() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  formatTime(ms: number): string {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const centiseconds = Math.floor((ms % 1000) / 10);

    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${centiseconds.toString().padStart(2, '0')}`;
  }

  getButtonIcon(): string {
    switch (this.gameState) {
      case 'ready': return '▶️';
      case 'playing': return '⏸️';
      case 'paused': return '▶️';
      default: return '▶️';
    }
  }

  getButtonText(): string {
    switch (this.gameState) {
      case 'ready': return 'Start Game';
      case 'playing': return 'Pause Game';
      case 'paused': return 'Resume Game';
      default: return 'Start Game';
    }
  }

  private loadStats() {
    const saved = localStorage.getItem('2048-stats');
    if (saved) {
      const stats = JSON.parse(saved);
      this.bestTime = stats.bestTime || 0;
      this.gamesPlayed = stats.gamesPlayed || 0;
    }
  }

  private saveStats() {
    const stats = {
      bestTime: this.bestTime,
      gamesPlayed: this.gamesPlayed
    };
    localStorage.setItem('2048-stats', JSON.stringify(stats));
  }

  // Public methods for game integration
  public completeGame() {
    if (this.gameState === 'playing') {
      this.gameState = 'ready';
      this.clearTimer();
      this.gamesPlayed++;

      if (this.bestTime === 0 || this.elapsedTime < this.bestTime) {
        this.bestTime = this.elapsedTime;
      }

      this.saveStats();
      this.elapsedTime = 0;
    }
  }

  public isPlaying(): boolean {
    return this.gameState === 'playing';
  }
}
