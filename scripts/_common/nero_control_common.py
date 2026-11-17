#!/usr/bin/env python3
"""独立机械臂使能、失能和软件急停恢复的公共模块，无位置运动命令。"""
import argparse
from dataclasses import dataclass, field
import math
import sys
import time

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
from .nero_local_read import ARMS, resolve_interfaces


@dataclass
class State:
    q: list
    velocity: list
    enabled: list
    stamp: float
    control: int
    fault: int
    error: int
    moving: int
    driver_faults: list = field(default_factory=list)


DRIVER_FAULT_FIELDS = ('voltage_too_low', 'motor_overheating', 'driver_overcurrent',
                       'driver_overheating', 'collision_status', 'driver_error_status', 'stall_status')


def describe_state(state):
    return (f'arm_status={state.fault}, err_code=0x{state.error:04X}, '
            f'ctrl_mode={state.control}, motion_status={state.moving}, '
            f'enabled={state.enabled}, 驱动故障={state.driver_faults or "无"}')


def fresh(message):
    stamp = getattr(message, 'timestamp', None)
    return isinstance(stamp, (int, float)) and math.isfinite(stamp) and -0.1 <= time.time()-stamp <= 0.5


def read_state(robot):
    if robot.has_comm_error():
        raise RuntimeError('SDK 报告 CAN 通信错误')
    angles, status = robot.get_joint_angles(), robot.get_arm_status()
    motors = [robot.get_motor_states(i) for i in range(1, 8)]
    drivers = [robot.get_driver_states(i) for i in range(1, 8)]
    if not all(fresh(m) for m in [angles, status, *motors, *drivers]):
        raise RuntimeError('七轴角度、速度、使能或状态反馈缺失/过期')
    q = list(angles.msg)
    velocity = [float(m.msg.velocity) for m in motors]
    positions = [float(m.msg.position) for m in motors]
    if len(q) != 7 or not all(math.isfinite(v) for v in [*q, *velocity, *positions]):
        raise RuntimeError('七轴反馈包含无效数值')
    if any(abs(a-b) > math.radians(1) for a, b in zip(q, positions)):
        raise RuntimeError('关节角与电机位置反馈不一致')
    faults = []
    for index, driver in enumerate(drivers, 1):
        for name in DRIVER_FAULT_FIELDS:
            value = getattr(driver.msg.foc_status, name, None)
            if value is None:
                raise RuntimeError(f'J{index} 缺少保护状态字段 {name}')
            if value:
                faults.append(f'J{index}.{name}')
    return State(q, velocity, [bool(d.msg.foc_status.driver_enable_status) for d in drivers],
                 float(angles.timestamp), int(status.msg.ctrl_mode), int(status.msg.arm_status),
                 int(status.msg.err_code), int(status.msg.motion_status), faults)


def power_feedback(robot):
    if robot.has_comm_error():
        raise RuntimeError('SDK 报告 CAN 通信错误')
    drivers = [robot.get_driver_states(i) for i in range(1, 8)]
    if not all(fresh(d) for d in drivers):
        raise RuntimeError('七轴使能反馈缺失或过期')
    return ([bool(d.msg.foc_status.driver_enable_status) for d in drivers],
            [float(d.timestamp) for d in drivers])


def wait_state(robot, timeout=5):
    deadline = time.monotonic()+timeout
    while True:
        try:
            return read_state(robot)
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def create_robot(side, factory=None):
    channel = resolve_interfaces()[side]
    cfg = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=NeroFW.V120,
                               interface='socketcan', channel=channel)
    return (factory or AgxArmFactory.create_arm)(cfg), cfg, channel


def switch_power(robot, cfg, enabled, timeout):
    deadline = time.monotonic()+timeout
    if enabled:
        initial = wait_state(robot, min(5, timeout))
        print('使能前诊断：', describe_state(initial), flush=True)
        disabled_brake = initial.fault == 6 and not any(initial.enabled)
        if initial.error or initial.driver_faults or (initial.fault != 0 and not disabled_brake):
            raise RuntimeError('拒绝使能：'+describe_state(initial))
        if initial.moving or any(abs(v) > math.radians(0.5) for v in initial.velocity):
            raise RuntimeError('机械臂未静止，拒绝使能')
        for i, q in enumerate(initial.q, 1):
            low, high = cfg['joint_limits'][f'joint{i}']
            if not low <= q <= high:
                raise RuntimeError(f'J{i} 当前角度超出 SDK 限位')
    else:
        try:
            flags, _ = power_feedback(robot)
            if not any(flags):
                return flags
        except RuntimeError:
            pass
    sent_at = time.time()
    last_error = '无完整新反馈'
    while time.monotonic() < deadline:
        if enabled:
            robot.enable(timeout=0.3)
        else:
            robot.disable(timeout=0.3)
        try:
            flags, stamps = power_feedback(robot)
            state = read_state(robot) if enabled else None
        except RuntimeError as error:
            last_error = str(error)
        else:
            if enabled and (state.error or state.driver_faults or state.fault not in (0, 6)):
                raise RuntimeError('使能中出现故障：'+describe_state(state))
            if all(flag == enabled for flag in flags) and all(stamp > sent_at for stamp in stamps):
                if not enabled or (state.fault == 0 and state.control == 1 and all(state.enabled)):
                    return flags
            last_error = describe_state(state) if enabled else f'当前使能状态 {flags}'
        time.sleep(0.05)
    raise RuntimeError('未确认七轴状态：'+last_error)


def run_power(args, enabled, factory=None):
    if not math.isfinite(args.timeout) or not 0.5 <= args.timeout <= 30:
        raise ValueError('--timeout 必须在 0.5～30 秒之间')
    sides = ('left', 'right') if args.arm == 'both' else (args.arm,)
    failures = []
    for side in sides:
        robot = None
        try:
            robot, cfg, channel = create_robot(side, factory)
            robot.connect()
            print(f'{side}: {channel}, USB={ARMS[side]}', flush=True)
            if args.execute:
                flags = switch_power(robot, cfg, enabled, args.timeout)
                print('已确认使能：' if enabled else '已确认失能：', flags)
            else:
                deadline = time.monotonic()+args.timeout
                while True:
                    try:
                        flags, _ = power_feedback(robot)
                        break
                    except RuntimeError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.05)
                print('当前七轴使能状态：', flags)
                try:
                    state = wait_state(robot, min(3, args.timeout))
                    print('控制器诊断：', describe_state(state))
                    print('关节角(deg)：', [round(math.degrees(q), 3) for q in state.q])
                except RuntimeError as error:
                    print('完整诊断暂不可用：', error)
                print('仅预览；添加 --execute 才实际请求切换。')
        except Exception as error:
            failures.append(f'{side}: {error}')
            print(f'{side} 未确认：{error}', file=sys.stderr)
            if enabled:
                break
        finally:
            if robot is not None:
                try:
                    robot.disconnect()
                except Exception as error:
                    failures.append(f'{side}: 关闭连接失败：{error}')
    if failures:
        raise RuntimeError('; '.join(failures))


def check_recovery_state(state, initial, limits, expected_estop=True):
    if state.error or state.driver_faults:
        raise RuntimeError('存在额外故障，拒绝恢复：'+describe_state(state))
    if expected_estop and state.fault != 1:
        raise RuntimeError('恢复前急停状态已改变')
    if state.moving or any(abs(v) > math.radians(0.5) for v in state.velocity):
        raise RuntimeError('机械臂未稳定')
    if any(abs(a-b) > math.radians(0.2) for a, b in zip(state.q, initial.q)):
        raise RuntimeError('恢复准备过程中姿态变化超过 0.2 度')
    for i, q in enumerate(state.q, 1):
        low, high = limits[f'joint{i}']
        if not low <= q <= high:
            raise RuntimeError(f'J{i} 超出 SDK 限位')


def run_estop_recovery(side, args, factory=None):
    if not math.isfinite(args.timeout) or not 1 <= args.timeout <= 30:
        raise ValueError('--timeout 必须在 1～30 秒之间')
    robot, cfg, channel = create_robot(side, factory)
    started = False
    try:
        robot.connect()
        initial = wait_state(robot, min(5, args.timeout))
        print(f'{side}: {channel}, USB={ARMS[side]}')
        print('恢复前：', describe_state(initial))
        if initial.error or initial.driver_faults:
            raise RuntimeError('存在额外故障，不作为单纯软件急停恢复')
        if initial.fault in (0, 6):
            print('当前未报告急停，保留原电机状态，未发送控制请求。')
            return
        if initial.fault != 1:
            raise RuntimeError('仅处理 arm_status=1 的软件急停')
        check_recovery_state(initial, initial, cfg['joint_limits'])
        if any(initial.enabled) and not all(initial.enabled):
            raise RuntimeError('处于部分使能，拒绝恢复；需独立检查')
        if not args.execute:
            print('仅诊断；实际恢复需 --execute；失能时还需 --enable-for-reset。')
            return
        if not all(initial.enabled):
            if not args.enable_for_reset:
                raise ValueError('当前失能；复位前短暂使能需明确添加 --enable-for-reset')
            started = True
            enable_sent = time.time()
            deadline = time.monotonic()+args.timeout
            while True:
                robot.enable(timeout=0.3)
                state = read_state(robot)
                check_recovery_state(state, initial, cfg['joint_limits'])
                flags, stamps = power_feedback(robot)
                if all(flags) and all(state.enabled) and all(s > enable_sent for s in stamps):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('未确认复位所需的七轴使能')
                time.sleep(0.05)
        state = read_state(robot)
        check_recovery_state(state, initial, cfg['joint_limits'])
        if state.control != 1 or not all(state.enabled):
            raise RuntimeError('复位前未确认 CAN 控制模式与七轴使能，停止恢复')
        print('发送一次 reset；复位可能撤去力矩，请确保机械臂有稳定支撑。', flush=True)
        started = True
        reset_sent = time.time()
        robot.reset()
        switch_power(robot, cfg, False, args.timeout)
        deadline = time.monotonic()+args.timeout
        while time.monotonic() < deadline:
            try:
                state = read_state(robot)
                flags, stamps = power_feedback(robot)
                status = robot.get_arm_status()
            except RuntimeError:
                time.sleep(0.05)
                continue
            if state.error or state.driver_faults or state.fault not in (0, 1, 6):
                raise RuntimeError('复位后仍有故障：'+describe_state(state))
            if (state.fault in (0, 6) and fresh(status) and status.timestamp > reset_sent
                    and state.stamp > reset_sent and not any(flags) and not any(state.enabled)
                    and all(s > reset_sent for s in stamps)):
                print('已确认解除软件急停，七轴保持失能；不自动继续使能或运动。')
                return
            time.sleep(0.05)
        raise RuntimeError('未确认解除急停且失能，不自动重复复位')
    except BaseException:
        if started:
            print('恢复未完成，尝试失能；不再复位或使能。', file=sys.stderr)
            try:
                switch_power(robot, cfg, False, min(3, args.timeout))
            except Exception as error:
                print('失能未确认：', error, file=sys.stderr)
        raise
    finally:
        robot.disconnect()


def power_main(enabled):
    parser = argparse.ArgumentParser(description='NERO 独立维护：使能/失能，默认只预览')
    parser.add_argument('--arm', choices=('left', 'right', 'both'), default='both')
    parser.add_argument('--timeout', type=float, default=6)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    try:
        run_power(args, enabled)
    except KeyboardInterrupt:
        print('已中断；请现场确认最终电机状态。', file=sys.stderr)
        return 130
    except Exception as error:
        print('操作未确认：', error, file=sys.stderr)
        return 1
    return 0


def recovery_main(side):
    parser = argparse.ArgumentParser(description=f'NERO {side} 软件急停恢复，默认只诊断')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--enable-for-reset', action='store_true')
    parser.add_argument('--timeout', type=float, default=6)
    args = parser.parse_args()
    try:
        run_estop_recovery(side, args)
    except KeyboardInterrupt:
        print('已中断；请现场确认最终电机状态。', file=sys.stderr)
        return 130
    except Exception as error:
        print('恢复未确认：', error, file=sys.stderr)
        return 1
    return 0
