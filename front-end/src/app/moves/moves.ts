import { Component,Input } from '@angular/core';

@Component({
  selector: 'app-moves',
  imports: [],
  templateUrl: './moves.html',
  styleUrl: './moves.scss'
})
export class Moves {
  @Input() count: number = 0;
}
