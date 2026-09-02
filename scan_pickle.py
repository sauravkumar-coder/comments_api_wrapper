"""Find all STACK_GLOBAL references in best_model.pkl."""
import sys, pickletools
sys.stdout.reconfigure(encoding='utf-8')

data = open('artifacts/best_model.pkl', 'rb').read()

globals_found = []
prev_strs = []

for opcode, arg, pos in pickletools.genops(data):
    if opcode.name in ('SHORT_BINUNICODE', 'BINUNICODE') and isinstance(arg, str):
        prev_strs.append(arg)
    elif opcode.name == 'STACK_GLOBAL':
        if len(prev_strs) >= 2:
            globals_found.append((prev_strs[-2], prev_strs[-1]))
        prev_strs = []
    # Don't clear on other opcodes

for m, c in sorted(set(globals_found)):
    print(f'{m}.{c}')
