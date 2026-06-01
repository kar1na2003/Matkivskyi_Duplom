# gdb_explorer.gdb — GDB batch script for PSE84 (PSVP-EXPLORER_B0) debug sessions.
#
# Derived from MTB launch config (.mtbLaunchConfigs/*.launch) — matches exactly
# how ModusToolbox debugs Explorer.
#
# Key PSE84 facts (from MTB config analysis):
#   1. ELF links at 0x083xxxxx = actual execution address via C-bus remap of NOR flash.
#      Load with: file <elf>  (NO offset needed — MTB does this too)
#   2. OpenOCD must use: gdb_breakpoint_override hard
#      Forces ALL breakpoints to hardware — required for XIP flash.
#   3. GDB must set: set mem inaccessible-by-default off
#      Allows reading NS peripheral registers even when halted in secure context.
#   4. Use: monitor reset_halt cm33_ns  (PSE84-specific OpenOCD command)
#      Halts specifically the NS core, walking through secure boot sequence.
#   5. After reset_halt cm33_ns: maintenance flush register-cache + mon gdb_sync
#      + thread apply all stepi  (synchronises GDB state — same as MTB)
#   6. set remotetimeout 500 — boot ROM sequence takes several seconds
#
# NOTE: 'target remote' and 'file <ELF>' are injected by make_app.sh via -ex flags
# before this script runs. Do NOT add them here.
#
# Usage:
#   source build/common/setenv.sh
#   GDB_SCRIPT=build/gdb_explorer.gdb \
#   bash build/make_app.sh APP=<app> TARGET=PSVP-EXPLORER_B0 CORE=CM33 IMAGE=non-secure \
#       APPTYPE=flash TOOLCHAIN=GCC_ARM CONFIG=Debug ACTION=debug
# ============================================================================

set pagination off
set remotetimeout 500
set mem inaccessible-by-default off

echo \n=== Halting NS core (MTB method: reset_halt cm33_ns) ===\n
monitor reset_halt cm33_ns
maintenance flush register-cache
mon gdb_sync
thread apply all stepi

echo \n=== Breaking at main ===\n
break main
continue

echo \n=== PC / Source ===\n
list
x/4i $pc

echo \n=== Core Registers ===\n
info registers r0 r1 r2 r3 r4 sp lr pc xpsr

echo \n=== 4x stepi ===\n
stepi
x/2i $pc
stepi
x/2i $pc
stepi
x/2i $pc
stepi
x/2i $pc

echo \n=== GPIO_PRT16 OUT @ 0x42810800 (bit7=LED_RED) ===\n
x/1xw 0x42810800

echo \n=== Fault Status Registers (all 0 = no fault) ===\n
x/1xw 0xE000ED28
x/1xw 0xE000ED2C
x/1xw 0xE000ED38

echo \n=== Backtrace ===\n
backtrace 20

echo \n=== Done ===\n
delete breakpoints
quit
