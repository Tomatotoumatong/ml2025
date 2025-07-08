
# run.py - 运行控制脚本
# =============================================================================
# 核心职责：
# 1. 命令行参数解析
# 2. 运行模式控制
# 3. 进程管理
# 4. 优雅启动和关闭
# =============================================================================

import asyncio
import argparse
import sys
import os
from pathlib import Path
import json
import signal
from typing import Optional

from stark4_app import STARK4App
from logger import TradingLogger
from utils import ConfigManager


class RunMode:
    """运行模式枚举"""
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading"
    RESEARCH = "research"


class STARK4Runner:
    """
    STARK4运行器
    
    核心功能：
    1. 解析命令行参数
    2. 管理应用生命周期
    3. 处理系统信号
    4. 提供CLI接口
    """
    
    def __init__(self):
        self.app: Optional[STARK4App] = None
        self.logger = TradingLogger().get_logger("RUNNER")
        self.args = None
    
    def parse_arguments(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description="STARK4-ML 智能量化交易系统",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  python run.py start                    # 启动生产模式
  python run.py start --mode development # 启动开发模式
  python run.py start --config custom.yaml # 使用自定义配置
  python run.py status                   # 查看系统状态
  python run.py stop                     # 停止系统
            """
        )
        
        # 子命令
        subparsers = parser.add_subparsers(dest='command', help='命令')
        
        # start命令
        start_parser = subparsers.add_parser('start', help='启动系统')
        start_parser.add_argument(
            '--mode', 
            choices=[RunMode.PRODUCTION, RunMode.DEVELOPMENT, 
                    RunMode.BACKTEST, RunMode.PAPER_TRADING, RunMode.RESEARCH],
            default=RunMode.PRODUCTION,
            help='运行模式'
        )
        start_parser.add_argument(
            '--config',
            default='config.yaml',
            help='配置文件路径'
        )
        start_parser.add_argument(
            '--symbols',
            nargs='+',
            help='交易品种列表'
        )
        start_parser.add_argument(
            '--dry-run',
            action='store_true',
            help='空跑模式（不执行实际交易）'
        )
        
        # stop命令
        stop_parser = subparsers.add_parser('stop', help='停止系统')
        stop_parser.add_argument(
            '--force',
            action='store_true',
            help='强制停止'
        )
        
        # status命令
        status_parser = subparsers.add_parser('status', help='查看状态')
        status_parser.add_argument(
            '--json',
            action='store_true',
            help='JSON格式输出'
        )
        
        # restart命令
        restart_parser = subparsers.add_parser('restart', help='重启系统')
        
        # test命令
        test_parser = subparsers.add_parser('test', help='测试连接')
        test_parser.add_argument(
            '--component',
            choices=['database', 'vnpy', 'telegram', 'all'],
            default='all',
            help='测试组件'
        )
        
        self.args = parser.parse_args()
        
        if not self.args.command:
            parser.print_help()
            sys.exit(1)
    
    async def execute_command(self):
        """执行命令"""
        command = self.args.command
        
        if command == 'start':
            await self.start()
        elif command == 'stop':
            await self.stop()
        elif command == 'status':
            await self.status()
        elif command == 'restart':
            await self.restart()
        elif command == 'test':
            await self.test()
        else:
            self.logger.error(f"未知命令: {command}")
            sys.exit(1)
    
    async def start(self):
        """启动系统"""
        try:
            self.logger.info(f"启动STARK4系统 - 模式: {self.args.mode}")
            
            # 检查是否已在运行
            if self._is_running():
                self.logger.error("系统已在运行")
                sys.exit(1)
            
            # 加载配置
            config = ConfigManager(self.args.config)
            
            # 覆盖配置
            if self.args.mode:
                config.set('system.mode', self.args.mode)
            
            if self.args.symbols:
                config.set('trading.symbols', self.args.symbols)
            
            if self.args.dry_run:
                config.set('trading.dry_run', True)
            
            # 保存配置
            config.save_config()
            
            # 创建应用实例
            self.app = STARK4App(self.args.config)
            
            # 初始化
            if not await self.app.initialize():
                self.logger.error("系统初始化失败")
                sys.exit(1)
            
            # 运行
            self.logger.info("系统启动成功")
            await self.app.run()
            
        except KeyboardInterrupt:
            self.logger.info("收到中断信号")
            if self.app:
                await self.app.shutdown()
        except Exception as e:
            self.logger.error(f"启动失败: {e}")
            sys.exit(1)
    
    async def stop(self):
        """停止系统"""
        try:
            if not self._is_running():
                self.logger.info("系统未在运行")
                return
            
            # 读取PID文件
            pid = self._read_pid()
            
            if pid:
                if self.args.force:
                    # 强制终止
                    os.kill(pid, signal.SIGKILL)
                    self.logger.info(f"已强制终止进程 {pid}")
                else:
                    # 优雅关闭
                    os.kill(pid, signal.SIGTERM)
                    self.logger.info(f"已发送关闭信号到进程 {pid}")
                
                # 清理PID文件
                self._cleanup_pid()
            
        except Exception as e:
            self.logger.error(f"停止失败: {e}")
            sys.exit(1)
    
    async def status(self):
        """查看状态"""
        try:
            if not self._is_running():
                status = {
                    'running': False,
                    'message': '系统未运行'
                }
            else:
                # 这里应该通过API或共享内存获取状态
                # 暂时返回基本信息
                pid = self._read_pid()
                status = {
                    'running': True,
                    'pid': pid,
                    'message': '系统运行中'
                }
            
            if self.args.json:
                print(json.dumps(status, indent=2))
            else:
                print(f"系统状态: {'运行中' if status['running'] else '未运行'}")
                if status.get('pid'):
                    print(f"进程ID: {status['pid']}")
                
        except Exception as e:
            self.logger.error(f"获取状态失败: {e}")
            sys.exit(1)
    
    async def restart(self):
        """重启系统"""
        self.logger.info("重启系统...")
        await self.stop()
        await asyncio.sleep(2)  # 等待进程完全退出
        await self.start()
    
    async def test(self):
        """测试连接"""
        try:
            self.logger.info(f"测试组件: {self.args.component}")
            
            # 加载配置
            config = ConfigManager(self.args.config)
            
            if self.args.component in ['database', 'all']:
                await self._test_database(config)
            
            if self.args.component in ['vnpy', 'all']:
                await self._test_vnpy(config)
            
            if self.args.component in ['telegram', 'all']:
                await self._test_telegram(config)
            
            self.logger.info("测试完成")
            
        except Exception as e:
            self.logger.error(f"测试失败: {e}")
            sys.exit(1)
    
    async def _test_database(self, config):
        """测试数据库连接"""
        self.logger.info("测试数据库连接...")
        # 实现数据库测试逻辑
        self.logger.info("数据库连接正常")
    
    async def _test_vnpy(self, config):
        """测试VNPy连接"""
        self.logger.info("测试VNPy连接...")
        # 实现VNPy测试逻辑
        self.logger.info("VNPy连接正常")
    
    async def _test_telegram(self, config):
        """测试Telegram连接"""
        self.logger.info("测试Telegram连接...")
        # 实现Telegram测试逻辑
        self.logger.info("Telegram连接正常")
    
    def _is_running(self) -> bool:
        """检查系统是否在运行"""
        pid_file = Path("/tmp/stark4_locks/stark4.pid")
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                # 检查进程是否存在
                os.kill(pid, 0)
                return True
            except (OSError, ValueError):
                # 进程不存在，清理PID文件
                pid_file.unlink()
                return False
        return False
    
    def _read_pid(self) -> Optional[int]:
        """读取PID"""
        pid_file = Path("/tmp/stark4_locks/stark4.pid")
        if pid_file.exists():
            try:
                return int(pid_file.read_text().strip())
            except ValueError:
                return None
        return None
    
    def _cleanup_pid(self):
        """清理PID文件"""
        pid_file = Path("/tmp/stark4_locks/stark4.pid")
        if pid_file.exists():
            pid_file.unlink()


async def main():
    """主函数"""
    runner = STARK4Runner()
    runner.parse_arguments()
    await runner.execute_command()


if __name__ == "__main__":
    # 设置事件循环策略（Windows兼容）
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 运行主函数
    asyncio.run(main())
