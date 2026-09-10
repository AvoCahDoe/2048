/** Snake / corner heuristic for watch mode (mirrors Python HeuristicAgent). */

const SNAKE = [
  [15, 14, 13, 12],
  [8, 9, 10, 11],
  [7, 6, 5, 4],
  [0, 1, 2, 3],
];

export type Dir = 'Up' | 'Right' | 'Down' | 'Left';

const DIRS: Dir[] = ['Up', 'Right', 'Down', 'Left'];

function clone(board: number[][]): number[][] {
  return board.map((r) => r.slice());
}

function slideLeft(row: number[]): { row: number[]; score: number } {
  const nonzero = row.filter((n) => n !== 0);
  const merged: number[] = [];
  let score = 0;
  for (let i = 0; i < nonzero.length; i++) {
    if (i + 1 < nonzero.length && nonzero[i] === nonzero[i + 1]) {
      const v = nonzero[i] * 2;
      merged.push(v);
      score += v;
      i++;
    } else {
      merged.push(nonzero[i]);
    }
  }
  while (merged.length < 4) merged.push(0);
  return { row: merged, score };
}

function transpose(board: number[][]): number[][] {
  const n = board.length;
  const out = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      out[i][j] = board[j][i];
    }
  }
  return out;
}

/** Apply direction; returns new board + score gained + whether moved. */
export function applyMove(
  board: number[][],
  dir: Dir
): { board: number[][]; score: number; moved: boolean } {
  let b = clone(board);
  let score = 0;
  if (dir === 'Left') {
    const next = b.map((row) => {
      const r = slideLeft(row);
      score += r.score;
      return r.row;
    });
    return { board: next, score, moved: JSON.stringify(next) !== JSON.stringify(b) };
  }
  if (dir === 'Right') {
    b = b.map((r) => r.slice().reverse());
    const res = applyMove(b, 'Left');
    return {
      board: res.board.map((r) => r.reverse()),
      score: res.score,
      moved: res.moved,
    };
  }
  if (dir === 'Up') {
    b = transpose(b);
    const res = applyMove(b, 'Left');
    return { board: transpose(res.board), score: res.score, moved: res.moved };
  }
  // Down
  b = transpose(b);
  const res = applyMove(b, 'Right');
  return { board: transpose(res.board), score: res.score, moved: res.moved };
}

function monoLine(line: number[]): number {
  let inc = 0;
  let dec = 0;
  for (let i = 0; i < 3; i++) {
    const a = line[i];
    const b = line[i + 1];
    if (a > b) dec += a - b;
    else if (b > a) inc += b - a;
  }
  return -Math.min(inc, dec);
}

export function evaluateBoard(board: number[][]): number {
  let empties = 0;
  let snake = 0;
  let smooth = 0;
  let mono = 0;
  let maxTile = 0;
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      const v = board[i][j];
      if (v === 0) empties++;
      else {
        snake += v * SNAKE[i][j];
        maxTile = Math.max(maxTile, v);
      }
      if (j + 1 < 4) smooth -= Math.abs(v - board[i][j + 1]);
      if (i + 1 < 4) smooth -= Math.abs(v - board[i + 1][j]);
    }
    mono += monoLine(board[i]);
    mono += monoLine([board[0][i], board[1][i], board[2][i], board[3][i]]);
  }
  return (
    2.7 * empties +
    snake +
    0.1 * smooth +
    mono +
    0.5 * Math.log2(maxTile + 1)
  );
}

export function chooseHeuristicMove(board: number[][]): Dir | null {
  let best = -Infinity;
  const candidates: Dir[] = [];
  for (const dir of DIRS) {
    const { board: next, score, moved } = applyMove(board, dir);
    if (!moved) continue;
    const s = evaluateBoard(next) + 0.1 * score;
    if (s > best + 1e-9) {
      best = s;
      candidates.length = 0;
      candidates.push(dir);
    } else if (Math.abs(s - best) < 1e-9) {
      candidates.push(dir);
    }
  }
  if (!candidates.length) return null;
  return candidates[Math.floor(Math.random() * candidates.length)];
}

export function hasLegalMoves(board: number[][]): boolean {
  for (const dir of DIRS) {
    if (applyMove(board, dir).moved) return true;
  }
  return false;
}

export function maxTile(board: number[][]): number {
  let m = 0;
  for (const row of board) for (const v of row) m = Math.max(m, v);
  return m;
}
