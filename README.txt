ModusMate bundle
================

host-side/
    Python host code (training, sweep, GUI, project_creator).
    Setup:
        cd host-side/host
        python3 -m venv .venv
        .venv/bin/pip install -e .
        .venv/bin/pip install kagglehub

board-side/ws-camera-imgproc-usb/
    ModusToolbox workspace with mtb_shared bundled.
    Build:
        cd board-side/ws-camera-imgproc-usb/camera-imgproc-usb
        make build -j

Flash a model:
    cd host-side
    python -m modusmate_host.models flash <name> --port <port> \
        --fw <abs path to board-side/ws-camera-imgproc-usb/camera-imgproc-usb>

Report workspace:
        All thesis/report materials are in report/:
            - report/test.tex + report/all_results_tables.tex
            - report/data/ (csv/xlsx inputs)
            - report/plots/ and report/images/
            - report/scripts/ (plot generation)

        Regenerate report plots:
                cd report
                python scripts/generate_report_plots.py
                python scripts/generate_family_plots.py
