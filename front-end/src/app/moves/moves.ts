import { CommonModule } from '@angular/common';
// moves.ts
import { Component, Input } from '@angular/core';

export interface Move {
  moveNumber: number;
  direction: string;
  scoreGained: number;
  timestamp: Date;
}

@Component({
  selector: 'app-moves',
  imports: [CommonModule],
  templateUrl: './moves.html',
  styleUrl: './moves.scss'

})
export class Moves {
  @Input() count: number = 0;
  @Input() moves: Move[] = [];

  trackByMove(index: number, move: Move): number {
    return move.moveNumber;
  }

  getDirectionIcon(direction: string): string {
    const icons: { [key: string]: string } = {
      'Left': '←',
      'Right': '→',
      'Up': '↑',
      'Down': '↓'
    };
    return icons[direction] || '?';
  }

  formatTime(timestamp: Date): string {
    return timestamp.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  }
}
