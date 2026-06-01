# gdb_default.gdb — GDB batch script for RRAM-based targets (Mercury, Atomic2).
#
# For XIP NOR flash targets use the platform-specific scripts:
#   PSVP-EXPLORER_B0 → build/gdb_explorer.gdb
#   PSVP-BOY4        → build/gdb_boy4.gdb
#
# Usage (via make_app.sh):
#   GDB_SCRIPT=build/gdb_default.gdb \
#   bash build/make_app.sh APP=blinky TARGET=PSVP-MERCURY CORE=CM33 \
#       APPTYPE=rram CONFIG=Debug TOOLCHAIN=GCC_ARM ACTION=debug
#
# OpenOCD is started automatically by make_app.sh before this script runs.
# Produces a structured capture of: registers, PC, backtrace, fault status regs.
# ============================================================================

set print pretty on
set pagination off
set confirm off

# ---- connection is already established by make_app.sh ----
# (target remote :3333 is handled before this script runs)
# We do a reset and halt to put the device in a known state.
monitor reset halt

echo \n=== Registers ===\n
info registers

echo \n=== Program Counter ===\n
print/x $pc
info symbol $pc

echo \n=== Backtrace ===\n
backtrace 20

echo \n=== Fault Status Registers (ARM Cortex-M) ===\n
# CFSR — Combined Fault Status Register (MemManage + BusFault + UsageFault)
print/x *(uint32_t*)0xE000ED28
# HFSR — HardFault Status Register
print/x *(uint32_t*)0xE000ED2C
# MMFAR — MemManage Fault Address Register
print/x *(uint32_t*)0xE000ED34
# BFAR — BusFault Address Register
print/x *(uint32_t*)0xE000ED38

echo \n=== Stack top (16 words) ===\n
x/16xw $sp

echo \n=== Done ===\n
quit
