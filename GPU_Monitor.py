import os
import json
import signal
import subprocess
import time
def get_GpuInfo(ip):
    """
    :param ip: host
    :return: gpu利用率, gpu内存占用率, gpu温度, gpu数量
    """

    utilization_list = []
    memory_usge_list = []
    temperature_list = []
    memory_used_list = []
    timeout_seconds = 30

    # gpu_cmd = 'ssh -o StrictHostKeyChecking=no %s gpustat --json' % ip  # 通过命令行执行gpustat --json
    gpu_cmd = 'gpustat --json'# 通过命令行执行gpustat --json

    gpu_info_dict = {}
    gpu_num = 0

    try:
        res = timeout_Popen(gpu_cmd, timeout_seconds)  # 超过30秒无返回信息,返回空值

        if res:
            res = res.stdout.read().decode()
            if not res:
                print('ssh %s 连接失败, 获取gpu信息失败' % ip)

            else:
                # gpu_info_dict = eval(res)
                gpu_info_dict = json.loads(res)  # str to json
                gpu_num = len(gpu_info_dict['gpus'])
    except:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    if gpu_info_dict:
        for i in gpu_info_dict['gpus']:
            utilization_gpu = float(i['utilization.gpu'])  # gpu利用率
            memory_used_gpu = round(100 * (i['memory.used'] / i['memory.total']), 2)  # gpu内存占用率
            memory_used = i['memory.used']
            
    
            memory_used_list.append(str(memory_used)) #内存使用量
            utilization_list.append(str(utilization_gpu))
            memory_usge_list.append(str(memory_used_gpu))
            temperature_list.append(str(i['temperature.gpu'])) # 温度

    else:
        print('{}: timeout > {}s, 获取gpu信息失败\n'.format(ip, timeout_seconds))
        utilization_list = ['-1']*4
        memory_usge_list = ['-1']*4
        temperature_list = ['-1']*4
    
    gpu_utilization = ','.join(utilization_list)
    gpu_memory = ','.join(memory_used_list) #内存使用量
    gpu_memory_utilization = ','.join(memory_usge_list)
    gpu_temperature = ','.join(temperature_list)

    return [gpu_utilization, gpu_memory, gpu_memory_utilization, gpu_temperature, gpu_num]


# 处理popen等待超时:
def timeout_Popen(cmd, timeout=30):
    start = time.time()
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while process.poll() is None:  # 是否结束
        time.sleep(0.2)
        now = time.time()
        if now - start >= timeout:
            os.kill(process.pid, signal.SIGKILL)

            # pid=-1 等待当前进程的all子进程, os.WNOHANG 没有子进程退出,
            os.waitpid(-1, os.WNOHANG)
            return None

    return process

if __name__ == "__main__":
    get_GpuInfo("127.0.0.1")