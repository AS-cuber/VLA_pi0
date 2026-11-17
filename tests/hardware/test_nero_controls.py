"""恢复后维护功能的离线回归检查，不连接硬件。"""
import contextlib
import io
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._common import nero_control_common as c
from scripts._common.nero_local_read import ARMS


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.state=c.State([0.0]*7,[0.0]*7,[False]*7,time.time(),1,0,0,0)
        self.robot=MagicMock()
        self.cfg={'joint_limits':{f'joint{i}':(-math.pi,math.pi) for i in range(1,8)}}
        self.stack=contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        def state(*args,**kwargs):
            self.state.stamp=time.time()
            return self.state
        self.stack.enter_context(patch.object(c,'wait_state',side_effect=state))
        self.stack.enter_context(patch.object(c,'read_state',side_effect=state))
        self.stack.enter_context(patch.object(c,'power_feedback',side_effect=lambda r:(self.state.enabled.copy(),[time.time()]*7)))
        self.stack.enter_context(patch.object(c,'create_robot',return_value=(self.robot,self.cfg,'can_test')))
        self.robot.enable.side_effect=lambda **kwargs:setattr(self.state,'enabled',[True]*7)
        self.robot.disable.side_effect=lambda **kwargs:setattr(self.state,'enabled',[False]*7)
        self.robot.get_arm_status.side_effect=lambda:SimpleNamespace(timestamp=time.time())

    def args(self,execute=False,enable_for_reset=False):
        return SimpleNamespace(arm='left',execute=execute,timeout=1,enable_for_reset=enable_for_reset)

    def test_preview_does_not_change_power(self):
        c.run_power(self.args(),True)
        self.robot.enable.assert_not_called();self.robot.disable.assert_not_called()
        self.robot.reset.assert_not_called();self.robot.disconnect.assert_called_once()

    def test_enable_success_checked(self):
        c.run_power(self.args(True),True)
        self.assertTrue(all(self.state.enabled))
        self.robot.enable.assert_called();self.robot.move_j.assert_not_called()

    def test_enable_refuses_estop(self):
        self.state.fault=1
        with self.assertRaises(RuntimeError):c.run_power(self.args(True),True)
        self.robot.enable.assert_not_called()

    def test_disable_success_checked(self):
        self.state.enabled=[True]*7
        c.run_power(self.args(True),False)
        self.assertFalse(any(self.state.enabled));self.robot.disable.assert_called()

    def test_already_disabled_no_resend(self):
        c.run_power(self.args(True),False)
        self.robot.disable.assert_not_called()

    def test_recovery_preview_no_control(self):
        self.state.fault=1
        c.run_estop_recovery('left',self.args())
        self.robot.enable.assert_not_called();self.robot.reset.assert_not_called()

    def test_recovery_needs_explicit_enable_permission(self):
        self.state.fault=1
        with self.assertRaises(ValueError):c.run_estop_recovery('left',self.args(True))
        self.robot.enable.assert_not_called();self.robot.reset.assert_not_called()

    def test_recovery_refuses_additional_fault(self):
        self.state.fault=1;self.state.error=1
        with self.assertRaises(RuntimeError):c.run_estop_recovery('left',self.args(True,True))
        self.robot.enable.assert_not_called();self.robot.reset.assert_not_called()

    def test_recovery_reset_once_and_remains_disabled(self):
        self.state.fault=1
        self.robot.reset.side_effect=lambda:setattr(self.state,'fault',6)
        c.run_estop_recovery('left',self.args(True,True))
        self.robot.reset.assert_called_once();self.assertFalse(any(self.state.enabled))
        self.robot.move_j.assert_not_called()

    def test_recovery_refuses_drift(self):
        initial=c.State([0.0]*7,[0.0]*7,[True]*7,time.time(),1,1,0,0)
        self.state.fault=1;self.state.q[0]=math.radians(1)
        with self.assertRaises(RuntimeError):c.check_recovery_state(self.state,initial,self.cfg['joint_limits'])

    def test_maintenance_identity_not_changed_to_shared_bus(self):
        self.assertEqual(ARMS['left'],'002B001D4648570C20303731')
        self.assertEqual(ARMS['right'],'0020002C4648570A20303731')


if __name__=='__main__':unittest.main()
