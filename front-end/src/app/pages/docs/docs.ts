import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-docs',
  standalone: true,
  imports: [RouterLink],
  templateUrl: './docs.html',
  styleUrl: './docs.scss',
})
export class DocsPage {}
