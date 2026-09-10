import { ControlsComponent } from './../controls/controls';
import { Moves, Move } from './../moves/moves';
import { DataAnalysis } from './../data-analysis/data-analysis';
import {
  Component,
  HostListener,
  OnDestroy,
  OnInit,
  ViewChild,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  chooseHeuristicMove,
  Dir,
  hasLegalMoves,
  maxTile,
} from './heuristic';

export type PlayMode = 'human' | 'watch';

@Component({
  selector: 'app-game',
  standalone: true,
  imports: [CommonModule, Moves, ControlsComponent, DataAnalysis],
  templateUrl: './game.html',
  styleUrls: ['./game.scss'],
})
export class GameComponent implements OnInit, OnDestroy {
  @ViewChild(ControlsComponent) controls?: ControlsComponent;

  gridSize = 4;
  board: number[][] = [];
  score = 0;
  moves = 0;
  moveHistory: Move[] = [];
  gameStarted = false;
  gamePaused = false;
  playMode: PlayMode = 'human';
  status: 'playing' | 'won' | 'lost' = 'playing';
  wonAcknowledged = false;

  private aiTimer: ReturnType<typeof setInterval> | null = null;
  private readonly aiIntervalMs = 180;

  ngOnInit(): void {
    this.initBoard();
  }

  ngOnDestroy(): void {
    this.stopAi();
  }

  initBoard(): void {
    this.board = Array.from({ length: this.gridSize }, () =>
      Array(this.gridSize).fill(0)
    );
    this.addRandomTile();
    this.addRandomTile();
    this.status = 'playing';
    this.wonAcknowledged = false;
  }

  addRandomTile(): void {
    const emptyCells: { x: number; y: number }[] = [];
    this.board.forEach((row, x) =>
      row.forEach((cell, y) => {
        if (cell === 0) emptyCells.push({ x, y });
      })
    );
    if (emptyCells.length === 0) return;
    const { x, y } = emptyCells[Math.floor(Math.random() * emptyCells.length)];
    this.board[x][y] = Math.random() < 0.9 ? 2 : 4;
  }

  setMode(mode: PlayMode): void {
    this.playMode = mode;
    if (mode === 'watch' && this.gameStarted && !this.gamePaused) {
      this.startAi();
    } else {
      this.stopAi();
    }
  }

  @HostListener('window:keydown', ['$event'])
  handleKey(event: KeyboardEvent): void {
    if (!this.gameStarted || this.gamePaused || this.status === 'lost') return;
    if (this.playMode !== 'human') return;
    if (this.status === 'won' && !this.wonAcknowledged) return;

    const map: Record<string, Dir> = {
      ArrowLeft: 'Left',
      ArrowRight: 'Right',
      ArrowUp: 'Up',
      ArrowDown: 'Down',
    };
    const direction = map[event.key];
    if (!direction) return;
    event.preventDefault();
    this.performMove(direction);
  }

  continueAfterWin(): void {
    this.wonAcknowledged = true;
    this.status = 'playing';
    if (this.playMode === 'watch') this.startAi();
  }

  performMove(direction: Dir): boolean {
    if (this.status === 'lost') return false;
    const previousScore = this.score;
    let moved = false;
    switch (direction) {
      case 'Left':
        moved = this.moveLeft();
        break;
      case 'Right':
        moved = this.moveRight();
        break;
      case 'Up':
        moved = this.moveUp();
        break;
      case 'Down':
        moved = this.moveDown();
        break;
    }
    if (!moved) return false;

    this.addRandomTile();
    this.moves++;
    this.moveHistory = [
      ...this.moveHistory,
      {
        moveNumber: this.moves,
        direction,
        scoreGained: this.score - previousScore,
        timestamp: new Date(),
      },
    ];

    if (maxTile(this.board) >= 2048 && !this.wonAcknowledged) {
      this.status = 'won';
      this.stopAi();
      this.controls?.completeGame();
      return true;
    }
    if (!hasLegalMoves(this.board)) {
      this.status = 'lost';
      this.stopAi();
      this.controls?.completeGame();
    }
    return true;
  }

  private startAi(): void {
    this.stopAi();
    this.aiTimer = setInterval(() => this.aiStep(), this.aiIntervalMs);
  }

  private stopAi(): void {
    if (this.aiTimer) {
      clearInterval(this.aiTimer);
      this.aiTimer = null;
    }
  }

  private aiStep(): void {
    if (!this.gameStarted || this.gamePaused || this.status !== 'playing') {
      this.stopAi();
      return;
    }
    const dir = chooseHeuristicMove(this.board);
    if (!dir) {
      this.status = 'lost';
      this.stopAi();
      this.controls?.completeGame();
      return;
    }
    this.performMove(dir);
  }

  moveLeft(): boolean {
    let moved = false;
    for (let row = 0; row < this.gridSize; row++) {
      const currentRow = this.board[row].filter((n) => n !== 0);
      for (let col = 0; col < currentRow.length - 1; col++) {
        if (currentRow[col] === currentRow[col + 1]) {
          currentRow[col] *= 2;
          this.score += currentRow[col];
          currentRow[col + 1] = 0;
        }
      }
      const newRow = currentRow.filter((n) => n !== 0);
      while (newRow.length < this.gridSize) newRow.push(0);
      if (this.board[row].join() !== newRow.join()) moved = true;
      this.board[row] = newRow;
    }
    return moved;
  }

  moveRight(): boolean {
    this.board = this.board.map((row) => row.reverse());
    const moved = this.moveLeft();
    this.board = this.board.map((row) => row.reverse());
    return moved;
  }

  moveUp(): boolean {
    this.transpose();
    const moved = this.moveLeft();
    this.transpose();
    return moved;
  }

  moveDown(): boolean {
    this.transpose();
    const moved = this.moveRight();
    this.transpose();
    return moved;
  }

  transpose(): void {
    const newBoard = Array.from({ length: this.gridSize }, () =>
      Array(this.gridSize).fill(0)
    );
    for (let i = 0; i < this.gridSize; i++) {
      for (let j = 0; j < this.gridSize; j++) {
        newBoard[i][j] = this.board[j][i];
      }
    }
    this.board = newBoard;
  }

  getTileClass(value: number): string {
    if (value === 0) return 'bg-black text-white';
    if (value === 2) return 'bg-purple-100 text-black';
    if (value === 4) return 'bg-purple-200 text-black';
    if (value === 8) return 'bg-purple-300 text-black';
    if (value === 16) return 'bg-purple-400 text-white';
    if (value === 32) return 'bg-purple-500 text-white';
    if (value === 64) return 'bg-purple-600 text-white';
    if (value === 128) return 'bg-purple-700 text-white';
    if (value === 256) return 'bg-purple-800 text-white';
    return 'bg-purple-900 text-white';
  }

  handleGameStart(): void {
    this.gameStarted = true;
    this.gamePaused = false;
    if (this.status === 'lost' || this.status === 'won') {
      // keep board if continuing; otherwise leave as-is
    }
    if (this.playMode === 'watch' && this.status === 'playing') {
      this.startAi();
    }
  }

  handleGamePause(): void {
    this.gamePaused = true;
    this.stopAi();
  }

  handleGameReset(): void {
    this.gameStarted = false;
    this.gamePaused = false;
    this.stopAi();
    this.initBoard();
    this.score = 0;
    this.moves = 0;
    this.moveHistory = [];
  }
}
