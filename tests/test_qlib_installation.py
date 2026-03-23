import qlib
from qlib.data import D
from qlib.constant import REG_CN

def test_qlib_installation():
    """测试qlib安装是否成功"""
    try:
        # 初始化
        qlib.init(mount_path="~/.qlib/qlib_data/cn_data", region=REG_CN)
        print("✓ Qlib初始化成功")
        
        # 测试数据访问
        calendar = D.calendar(start_time='2020-01-01', end_time='2020-01-10', freq='day')
        print(f"✓ 数据访问成功，交易日历长度: {len(calendar)}")
        
        # 测试特征获取
        instruments = ['SH600000']
        fields = ['$close', '$volume']
        data = D.features(instruments, fields, start_time='2020-01-01', end_time='2020-01-10', freq='day')
        print(f"✓ 特征获取成功，数据形状: {data.shape}")
        
        print("🎉 Qlib环境配置成功！")
        return True
        
    except Exception as e:
        print(f"❌ 环境配置失败: {e}")
        return False

if __name__ == "__main__":
    test_qlib_installation()
