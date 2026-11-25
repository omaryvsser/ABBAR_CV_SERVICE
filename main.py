#!/usr/bin/env python3
"""
AABAR - Computer Vision Service | Phase 2 Main Entry Point.

Runs the real-time webcam gaze feature extraction and tracking pipeline.
Provides normalized iris positions, continuous eye openness, 3D head pose,
relative gaze direction estimation, adaptive EMA smoothing, and structured JSON output.
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import time
import cv2

# Ensure cv-service root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure MPLCONFIGDIR is set to writable directory
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_aabar"

from src.camera.camera_manager import CameraInitializationError
from src.config import AppConfig, load_config
from src.pipeline import CVPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AABAR-CV")


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(
        description="AABAR CV Service - Gaze Feature Extraction & Tracking Pipeline (Phase 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera-index", "-c",
        type=int,
        default=None,
        help="Camera device index (default: 0 or AABAR_CAMERA_INDEX from .env)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Frame capture width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Frame capture height (default: 720)",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=None,
        help="Target processing frame rate (default: 30)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Run in headless mode without opening GUI visualization window",
    )
    parser.add_argument(
        "--draw-mesh",
        action="store_true",
        help="Render complete facial mesh connections in addition to eyes and irises",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom path to MediaPipe face_landmarker.task model",
    )
    parser.add_argument(
        "--mock-camera",
        action="store_true",
        help="Use synthetic frames instead of physical webcam (ideal for testing without camera hardware/permissions)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Exit automatically after processing this number of frames (useful for CI/testing)",
    )
    parser.add_argument(
        "--unmirrored",
        action="store_true",
        help="Disable selfie mirror preview in debug window and display raw un-mirrored camera feed",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print structured JSON output for processed frames to stdout",
    )
    parser.add_argument(
        "--record-data",
        action="store_true",
        help="Automatically record diagnostic gaze tracking telemetry to CSV on startup",
    )
    parser.add_argument(
        "--skip-auto-calibration",
        action="store_true",
        help="Skip automatic neutral resting baseline calibration on pipeline startup",
    )
    parser.add_argument(
        "--calibrate-screen",
        action="store_true",
        help="Initiate 9-point screen gaze calibration on startup (auto-queues validation unless --calibration-only)",
    )
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Run calibration only without queuing automatic validation afterward",
    )
    parser.add_argument(
        "--validate-screen",
        action="store_true",
        help="Initiate screen gaze accuracy validation on startup (requires calibrated model or --profile)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path to pre-existing calibration profile JSON to load on startup",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=None,
        help="Display screen width in pixels (default: 1920)",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=None,
        help="Display screen height in pixels (default: 1080)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["auto", "polynomial", "linear", "compact_linear", "compact_polynomial"],
        default=None,
        help="Regression algorithm to use for gaze-to-screen mapping (default: auto)",
    )
    parser.add_argument(
        "--validation-repeats",
        type=int,
        default=None,
        help="Number of trials per target during validation (default: 1)",
    )
    parser.add_argument(
        "--randomize-val-targets",
        action="store_true",
        help="Randomize validation target presentation order to avoid sequential bias",
    )
    parser.add_argument(
        "--target-radius-px",
        type=float,
        default=None,
        help="Optional validation circular hit boundary radius in screen pixels (default: rendered 22px circle)",
    )
    parser.add_argument(
        "--target-width-px",
        type=float,
        default=None,
        help="Optional AAC button hit boundary width in screen pixels (pair with --target-height-px)",
    )
    parser.add_argument(
        "--target-height-px",
        type=float,
        default=None,
        help="Optional AAC button hit boundary height in screen pixels (pair with --target-width-px)",
    )
    parser.add_argument(
        "--dwell",
        action="store_true",
        help="Enable Phase 4 Dwell Selection Engine on startup",
    )
    parser.add_argument(
        "--dwell-time",
        type=int,
        default=None,
        help="Dwell duration in milliseconds before selection triggers (default: 1200)",
    )
    parser.add_argument(
        "--cooldown-time",
        type=int,
        default=None,
        help="Post-selection refractory cooldown in milliseconds (default: 800)",
    )
    parser.add_argument(
        "--dwell-demo",
        action="store_true",
        help="Launch interactive Phase 4 Dwell Selection manual testing mode with AAC target grid",
    )
    parser.add_argument(
        "--compare-experiments",
        type=str,
        nargs="+",
        default=None,
        help="Compare two or more experiment directory paths and exit",
    )
    parser.add_argument(
        "--compare-all-experiments",
        action="store_true",
        help="Scan data/experiments and print comparison table of all sessions",
    )
    parser.add_argument(
        "--sync-tracker",
        action="store_true",
        help="Scan all historical & recent experiments, recalculate metrics from raw trials, and update AABAR_Experiments_Tracker.xlsx",
    )
    parser.add_argument(
        "--show-results",
        type=str,
        nargs="?",
        const="latest",
        default=None,
        help="Display prominent accuracy results summary for 'latest' or a specific Experiment ID",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution loop for AABAR CV Service demo."""
    args = parse_args()

    # Handle central tracker sync and results inspection modes
    if args.sync_tracker or args.show_results:
        from src.core.experiment_tracker import format_experiment_results_summary, load_tracker_dataset, update_central_tracker
        cfg_tmp = load_config()
        exp_dir = Path(cfg_tmp.screen.experiments_dir)
        prof_dir = Path(cfg_tmp.screen.profiles_dir)
        if args.sync_tracker:
            sync_res = update_central_tracker(experiments_dir=exp_dir, profiles_dir=prof_dir)
            ds = sync_res["dataset"]
            t_path = sync_res["tracker_path"]
            print("\n" + "=" * 72)
            print(" AABAR CV — Central Experiment Tracker Synchronized")
            print("=" * 72)
            print(f" Workbook:                 {t_path}")
            print(f" Recorded Experiments:     {len(ds['experiments'])} (Empty startup dirs skipped: {len(ds['empty_dirs'])})")
            print(f" Completed Validation:     {ds['completed_validation_count']}")
            print(f" Missing/Incomplete Val:   {ds['unvalidated_count']}")
            print("=" * 72)
            for exp_rec in ds["experiments"]:
                print(format_experiment_results_summary(exp_rec, tracker_path=t_path))
        else:
            ds = load_tracker_dataset(experiments_dir=exp_dir, profiles_dir=prof_dir)
            t_path = exp_dir / "AABAR_Experiments_Tracker.xlsx"
            exps = ds["experiments"]
            if not exps:
                print("\nNo recorded experiments found under data/experiments/.\n")
                return 0
            if args.show_results == "latest":
                target_exp = exps[-1]
            else:
                matches = [e for e in exps if e["experiment_id"] == args.show_results]
                if not matches:
                    print(f"\nExperiment ID '{args.show_results}' not found in {exp_dir}.\n")
                    return 1
                target_exp = matches[0]
            print("\n" + format_experiment_results_summary(target_exp, tracker_path=t_path if t_path.exists() else None) + "\n")
        return 0

    # Handle experiment comparison CLI modes
    if args.compare_experiments or args.compare_all_experiments:
        from src.core.experiment_logger import compare_experiments
        if args.compare_all_experiments:
            base_dir = Path("data/experiments")
            target_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()]) if base_dir.exists() else []
        else:
            target_dirs = [Path(p) for p in args.compare_experiments]

        print("\n" + "=" * 70)
        print(" AABAR CV - Experiment Accuracy Benchmark Comparison")
        print("=" * 70)
        print(compare_experiments(target_dirs))
        print("=" * 70 + "\n")
        return 0

    config = load_config()

    # Apply command-line overrides
    if args.camera_index is not None:
        config.camera.camera_index = args.camera_index
    if args.width is not None:
        config.camera.frame_width = args.width
    if args.height is not None:
        config.camera.frame_height = args.height
    if args.target_fps is not None:
        config.camera.target_fps = args.target_fps
    if args.no_viz:
        config.visualization.enabled = False
    if args.unmirrored:
        config.visualization.mirror_display = False
    if args.draw_mesh:
        config.visualization.draw_face_mesh = True
    if args.model_path is not None:
        config.landmarker.model_path = args.model_path
    if args.skip_auto_calibration:
        config.gaze.auto_calibrate_on_start = False
    if args.profile is not None:
        config.screen.active_profile_path = args.profile
    if args.screen_width is not None:
        config.screen.screen_width = args.screen_width
    if args.screen_height is not None:
        config.screen.screen_height = args.screen_height
    if args.model_type is not None:
        config.screen.model_type = args.model_type
    if args.validation_repeats is not None:
        config.screen.validation_repeats = args.validation_repeats
    if args.randomize_val_targets:
        config.screen.randomize_validation = True
    if args.target_radius_px is not None:
        config.screen.validation_target_radius_px = args.target_radius_px
    if args.target_width_px is not None:
        config.screen.validation_target_width_px = args.target_width_px
    if args.target_height_px is not None:
        config.screen.validation_target_height_px = args.target_height_px
    if args.dwell or args.dwell_demo:
        config.dwell.enabled = True
    if args.dwell_time is not None:
        config.dwell.dwell_duration_ms = args.dwell_time
    if args.cooldown_time is not None:
        config.dwell.cooldown_duration_ms = args.cooldown_time

    model_file = Path(config.landmarker.model_path)
    if not model_file.exists():
        logger.error(
            f"Pretrained model bundle not found: {model_file}\n"
            f"Run `python download_model.py` to download the official MediaPipe model bundle."
        )
        return 1

    print("\n" + "=" * 65)
    print(" AABAR - Computer Vision Service | Phase 2 Gaze Tracking")
    print(" Gaze Feature Extraction, Head Pose & Direction for Arabic AAC")
    print("=" * 65)
    print(f" Camera Index:     {config.camera.camera_index}")
    print(f" Target Resolution: {config.camera.frame_width}x{config.camera.frame_height} @ {config.camera.target_fps} FPS")
    print(f" Model Bundle:     {model_file}")
    print(f" GUI Display:      {'Enabled' if config.visualization.enabled else 'Disabled (Headless)'}")
    if config.visualization.enabled:
        print(f" Preview Mode:     {'Selfie Mirror (Flipped)' if config.visualization.mirror_display else 'Unmirrored (Raw Camera)'}")
    dwell_status = (
        f"Enabled ({config.dwell.dwell_duration_ms}ms dwell, {config.dwell.cooldown_duration_ms}ms cooldown)"
        if config.dwell.enabled
        else "Disabled (Press 'd' to toggle)"
    )
    print(f" Dwell Engine:     {dwell_status}")
    print(" Controls:         's' = Calib+Val | 't' = Val | 'd' = Dwell | '1'-'4' = AAC Targets | 'x' = Reset Dwell | 'q'/ESC = Exit")
    print("=" * 65 + "\n")

    try:
        pipeline = CVPipeline(config=config, use_mock_camera=args.mock_camera)
        pipeline.start()
        if pipeline.dwell_engine is not None:
            pipeline.dwell_engine.on_selection(
                lambda evt: logger.info(
                    f"[DWELL SELECTION] Target '{evt.target_label}' ({evt.target_id}) triggered! "
                    f"Gaze=({evt.gaze_x:.3f}, {evt.gaze_y:.3f}) Duration={evt.dwell_duration_ms}ms"
                )
            )
        if args.record_data:
            rec_file = pipeline.start_data_logging()
            logger.info(f"Started diagnostic data recording on launch: {rec_file}")
        if args.calibrate_screen:
            pipeline.start_screen_calibration()
            if not args.calibration_only:
                pipeline.queue_screen_validation()
        elif args.validate_screen:
            if pipeline.active_mapper is not None:
                pipeline.start_screen_validation()
            else:
                logger.warning("Cannot start screen validation on startup: No profile loaded.")
    except CameraInitializationError as e:
        logger.error(f"Camera Initialization Failed: {e}")
        print("\n[HINT] Ensure your webcam is connected and terminal has Camera permissions in macOS.")
        print("[HINT] You can also run with `--mock-camera` to test the pipeline without physical camera hardware.")
        return 1
    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}", exc_info=True)
        return 1

    window_name = config.visualization.window_name
    target_frame_time = 1.0 / config.camera.target_fps if config.camera.target_fps > 0 else 0.0

    total_frames = 0
    t_start = time.time()

    try:
        while True:
            cycle_start = time.perf_counter()

            output = pipeline.process_next_frame()
            if output is None:
                # Brief pause before retrying
                time.sleep(0.005)
                continue

            total_frames += 1

            if args.max_frames and total_frames >= args.max_frames:
                logger.info(f"Reached requested limit of {args.max_frames} frames. Shutting down...")
                break

            # Optional structured JSON printing
            if args.print_json and total_frames % 15 == 0:
                print(output.gaze_frame_result.to_json(indent=2))

            # Log periodically to console in headless mode or every 60 frames
            if total_frames % 60 == 0:
                has_face = output.detection_result.has_face
                fps = output.fps_metrics.fps
                inf_ms = output.detection_result.inference_time_ms
                direction = output.gaze_direction.value
                conf = int(output.eye_features.confidence * 100)
                state_name = output.tracking_state.name
                pose_str = (
                    f"Pose(Y:{output.head_pose.yaw:+.1f}, P:{output.head_pose.pitch:+.1f}, R:{output.head_pose.roll:+.1f})"
                    if output.head_pose and output.head_pose.is_valid
                    else "Pose: N/A"
                )
                if has_face and output.eye_features.is_valid:
                    raw_h = f"{output.eye_features.combined_horizontal_ratio:.2f}" if output.eye_features.combined_horizontal_ratio is not None else "--"
                    raw_v = f"{output.eye_features.combined_vertical_ratio:.2f}" if output.eye_features.combined_vertical_ratio is not None else "--"
                    sm_h = f"{output.smoothed_x:.2f}" if output.smoothed_x is not None else "--"
                    sm_v = f"{output.smoothed_y:.2f}" if output.smoothed_y is not None else "--"
                    b_str = (
                        f"Base:({output.baseline.neutral_x:.2f},{output.baseline.neutral_y:.2f})"
                        if output.baseline and output.baseline.is_calibrated
                        else "Base: N/A"
                    )
                    d_x = output.gaze_frame_result.gaze.delta_x
                    d_y = output.gaze_frame_result.gaze.delta_y
                    dev_str = f"Dev:({d_x:+.2f},{d_y:+.2f})" if d_x is not None and d_y is not None else "Dev: N/A"
                    logger.info(
                        f"[Frame {total_frames:5d}] FPS: {fps:4.1f} | Latency: {inf_ms:4.1f}ms | "
                        f"GAZE: {direction:<7} | State: [{state_name}] | {b_str} | {dev_str} | {pose_str}"
                    )
                else:
                    logger.info(
                        f"[Frame {total_frames:5d}] FPS: {fps:4.1f} | Latency: {inf_ms:4.1f}ms | State: [{state_name}]"
                    )

            # Display GUI window if enabled
            if config.visualization.enabled and output.annotated_frame is not None:
                cv2.imshow(window_name, output.annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                    logger.info("Exit key pressed by user. Shutting down...")
                    break
                elif key in (ord("c"), ord("C")):
                    logger.info("Recalibration requested by user via keypress ('c').")
                    pipeline.recalibrate()
                elif key in (ord("v"), ord("V")):
                    logger.info("Vertical-only recalibration requested by user via keypress ('v').")
                    pipeline.recalibrate_vertical()
                elif key in (ord("s"), ord("S")):
                    if pipeline.screen_calibrator.is_active:
                        logger.info("Cancelling 9-point screen calibration via keypress ('s').")
                        pipeline.cancel_screen_calibration()
                    else:
                        logger.info("Starting 9-point screen calibration + independent validation via keypress ('s').")
                        pipeline.start_screen_calibration()
                        pipeline.queue_screen_validation()
                elif key in (ord("t"), ord("T")):
                    if pipeline.screen_validator.is_active:
                        logger.info("Cancelling screen validation via keypress ('t').")
                        pipeline.cancel_screen_validation()
                    elif pipeline.active_mapper is not None:
                        reuse_exp = pipeline.has_unvalidated_calibration_session()
                        logger.info(
                            f"Starting screen validation trial via keypress ('t', reuse_active_experiment={reuse_exp})."
                        )
                        pipeline.start_screen_validation(reuse_active_experiment=reuse_exp)
                    else:
                        logger.warning("Cannot validate: No screen calibration model available. Press 's' first.")
                elif key in (ord("y"), ord("Y")):
                    if pipeline.retry_failed_target():
                        logger.info("Retrying failed calibration/validation target via keypress ('y').")
                elif key in (ord("k"), ord("K")):
                    if pipeline.skip_failed_target():
                        logger.info("Skipped failed validation target via keypress ('k').")
                elif key in (ord("p"), ord("P")):
                    saved = pipeline.save_calibration_profile()
                    if saved:
                        logger.info(f"Saved active calibration profile to: {saved}")
                elif key in (ord("r"), ord("R")):
                    is_rec = pipeline.toggle_data_logging()
                    status_str = "STARTED" if is_rec else "STOPPED"
                    logger.info(f"Diagnostic telemetry recording {status_str} via keypress ('r').")
                elif key in (ord("d"), ord("D")):
                    is_en = pipeline.toggle_dwell_selection()
                    logger.info(f"Dwell Selection Engine toggled: {'ENABLED' if is_en else 'DISABLED'}")
                elif key in (ord("x"), ord("X")):
                    if pipeline.dwell_engine is not None:
                        pipeline.dwell_engine.reset()
                        logger.info("Dwell Engine manually reset to IDLE.")
                elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                    idx = chr(key)
                    tid = f"target_{idx}"
                    if pipeline.dwell_engine is not None:
                        tgt = pipeline.dwell_engine.target_manager.get_target(tid)
                        if tgt is not None:
                            tgt.enabled = not tgt.enabled
                            logger.info(f"Target '{tgt.label}' ({tid}) enabled state: {tgt.enabled}")

            # CPU throttle to respect target frame rate
            elapsed = time.perf_counter() - cycle_start
            sleep_needed = target_frame_time - elapsed
            if sleep_needed > 0.001:
                time.sleep(sleep_needed)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping pipeline...")
    finally:
        pipeline.close()
        cv2.destroyAllWindows()

    total_elapsed = time.time() - t_start
    avg_fps = total_frames / total_elapsed if total_elapsed > 0 else 0.0
    print("\n" + "=" * 65)
    print(" Session Summary")
    print(f" Total Frames:      {total_frames}")
    print(f" Elapsed Duration:  {total_elapsed:.2f} seconds")
    print(f" Average FPS:       {avg_fps:.1f} FPS")
    print("=" * 65 + "\n")

    if getattr(pipeline, "last_experiment_dir", None) is not None:
        try:
            from src.core.experiment_tracker import format_experiment_results_summary, load_tracker_dataset
            exp_root = Path(config.screen.experiments_dir)
            prof_root = Path(config.screen.profiles_dir)
            ds = load_tracker_dataset(experiments_dir=exp_root, profiles_dir=prof_root)
            matches = [e for e in ds["experiments"] if Path(e["path"]) == pipeline.last_experiment_dir]
            if matches:
                print(
                    format_experiment_results_summary(
                        matches[0],
                        tracker_path=getattr(pipeline, "last_tracker_path", None)
                        or (exp_root / "AABAR_Experiments_Tracker.xlsx"),
                    )
                    + "\n"
                )
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
