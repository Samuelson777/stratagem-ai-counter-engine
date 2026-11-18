import { setup, assign } from 'xstate';

export type GameModeId = 1 | 2 | 3 | 4 | 5 | 6;

export interface ArchetypeBelief {
  id: string;
  name: string;
  posterior: number;
  eloCondition: number;
}

export interface GameMachineContext {
  activeMode: GameModeId;
  plyCount: number;
  fen: string;
  activeTurn: 'w' | 'b';
  sparringElo: number;
  dchExploitBlend: number;
  mustAttack: boolean;
  canStop: boolean;
  activeTerritoryActor: number;
  beliefs: ArchetypeBelief[];
  revealedCounterPlan: string;
}

export type GameMachineEvent =
  | { type: 'SWITCH_MODE'; mode: GameModeId }
  | { type: 'SELECT_MOVE'; from: [number, number]; to: [number, number]; thinkTimeSec: number }
  | { type: 'AI_COUNTER_READY'; nextFen: string; counterPlan: string; updatedBeliefs: ArchetypeBelief[] }
  | { type: 'FLOOR_ATTACK_DEFENDER'; defenderId: number; attackerWon: boolean }
  | { type: 'FLOOR_PASS_TO_RANDOMIZER' }
  | { type: 'SET_DCH_BLEND'; weight: number }
  | { type: 'RESET_MATCH' };

export const stratagemStateMachine = setup({
  types: {
    context: {} as GameMachineContext,
    events: {} as GameMachineEvent
  },
  guards: {
    canPassToRandomizer: ({ context }) => !context.mustAttack && context.canStop,
    mustExecuteAttack: ({ context }) => context.mustAttack
  },
  actions: {
    abortPendingWorkers: () => {
      // Deterministically cancels in-flight MCTS rollouts on mode switch
      window.dispatchEvent(new CustomEvent('STRATAGEM_ABORT_WORKERS'));
    }
  }
}).createMachine({
  id: 'StratagemCounterEngine',
  initial: 'awaitingHumanInput',
  context: {
    activeMode: 1,
    plyCount: 0,
    fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    activeTurn: 'w',
    sparringElo: 1600,
    dchExploitBlend: 0.72,
    mustAttack: true,
    canStop: false,
    activeTerritoryActor: 0,
    beliefs: [
      { id: 'kingside_aggressor', name: 'Kingside Gambit Aggressor', posterior: 0.25, eloCondition: 1650 },
      { id: 'positional_squeeze', name: 'Positional Squeeze', posterior: 0.25, eloCondition: 1950 },
      { id: 'tactical_improviser', name: 'Tactical Improviser', posterior: 0.25, eloCondition: 1450 },
      { id: 'hypermodern_flank', name: 'Hypermodern Flank', posterior: 0.25, eloCondition: 1800 }
    ],
    revealedCounterPlan: 'Awaiting initial attack telemetry...'
  },
  on: {
    SWITCH_MODE: {
      target: '.awaitingHumanInput',
      actions: [
        'abortPendingWorkers',
        assign({ activeMode: ({ event }) => event.mode })
      ]
    },
    SET_DCH_BLEND: {
      actions: assign({ dchExploitBlend: ({ event }) => event.weight })
    },
    RESET_MATCH: {
      target: '.awaitingHumanInput',
      actions: [
        'abortPendingWorkers',
        assign({
          plyCount: 0,
          activeTurn: 'w',
          mustAttack: true,
          canStop: false
        })
      ]
    }
  },
  states: {
    awaitingHumanInput: {
      on: {
        SELECT_MOVE: {
          target: 'synthesizingCounterStrategy',
          actions: assign({
            plyCount: ({ context }) => context.plyCount + 1,
            activeTurn: ({ context }) => (context.activeTurn === 'w' ? 'b' : 'w')
          })
        },
        FLOOR_ATTACK_DEFENDER: {
          target: 'awaitingHumanInput',
          actions: assign({
            mustAttack: false,
            canStop: true
          })
        },
        FLOOR_PASS_TO_RANDOMIZER: {
          guard: 'canPassToRandomizer',
          target: 'awaitingHumanInput',
          actions: assign({
            mustAttack: true,
            canStop: false
          })
        }
      }
    },
    synthesizingCounterStrategy: {
      on: {
        AI_COUNTER_READY: {
          target: 'awaitingHumanInput',
          actions: assign({
            fen: ({ event }) => event.nextFen,
            revealedCounterPlan: ({ event }) => event.counterPlan,
            beliefs: ({ event }) => event.updatedBeliefs,
            plyCount: ({ context }) => context.plyCount + 1,
            activeTurn: ({ context }) => (context.activeTurn === 'w' ? 'b' : 'w')
          })
        }
      }
    }
  }
});