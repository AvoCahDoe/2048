// game.ts
import { Moves, Move } from './../moves/moves';
import { Component, HostListener, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-game',
  standalone: true,
  imports: [CommonModule, Moves],
  templateUrl: './game.html',
  styleUrls: ['./game.scss']
})
export class GameComponent implements OnInit {
  gridSize = 4;
  board: number[][] = [];
  score = 0;
  moves = 0;
  moveHistory: Move[] = [];

  ngOnInit(): void {
    this.initBoard();
  }

  initBoard(): void {
    this.board = Array.from({ length: this.gridSize }, () =>
      Array(this.gridSize).fill(0)
    );
    this.addRandomTile();
    this.addRandomTile();
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

  @HostListener('window:keydown', ['$event'])
  handleKey(event: KeyboardEvent): void {
    let moved = false;
    let direction = '';
    const previousScore = this.score;

    switch (event.key) {
      case 'ArrowLeft':
        moved = this.moveLeft();
        direction = 'Left';
        break;
      case 'ArrowRight':
        moved = this.moveRight();
        direction = 'Right';
        break;
      case 'ArrowUp':
        moved = this.moveUp();
        direction = 'Up';
        break;
      case 'ArrowDown':
        moved = this.moveDown();
        direction = 'Down';
        break;
    }

    if (moved) {
      this.addRandomTile();
      this.moves++;

      const scoreGained = this.score - previousScore;
      const move: Move = {
        moveNumber: this.moves,
        direction: direction,
        scoreGained: scoreGained,
        timestamp: new Date()
      };

      this.moveHistory.push(move); // Add to end for chronological order
    }
  }

  moveLeft(): boolean {
    let moved = false;
    for (let row = 0; row < this.gridSize; row++) {
      const currentRow = this.board[row].filter(n => n !== 0);
      for (let col = 0; col < currentRow.length - 1; col++) {
        if (currentRow[col] === currentRow[col + 1]) {
          currentRow[col] *= 2;
          this.score += currentRow[col];
          currentRow[col + 1] = 0;
        }
      }
      const newRow = currentRow.filter(n => n !== 0);
      while (newRow.length < this.gridSize) newRow.push(0);
      if (this.board[row].join() !== newRow.join()) moved = true;
      this.board[row] = newRow;
    }
    return moved;
  }

  moveRight(): boolean {
    this.board = this.board.map(row => row.reverse());
    const moved = this.moveLeft();
    this.board = this.board.map(row => row.reverse());
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
}
