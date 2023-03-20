from gym_mhop.envs.Maps import MapGenParms


class GlobalConfig:
    # ======= exp config =============

    TOTAL_HOPS = 3
    gen_parms = MapGenParms()
    gen_parms.K = 1
    gen_parms.MAX_M = 100
    gen_parms.MIN_M = 9
    gen_parms.lambda_o = 3
    gen_parms.lambda_p = 3

    TOTAL_ENVS = 1000
    ENV_RUNS = 1

    MAX_M = 100  # mainly for padding env observation
    G_CLIP = 0.2
    lr = 0.1**4
    gamma = 0.8
    gae_lambda = 0.8

    update_epochs = 4
    num_minibatches = 1
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.01
    max_grad_norm = 0.5
    vf_coef = 0.5

    USE_GAE = True
    EXPOSE_ALL = True

    # =======================
    # grand_server_ip = '81.71.146.174'
    GRAND_SERVER_IP = "127.0.0.1"
    # ftp_host = '81.71.146.174'
    ftp_host = "127.0.0.1"
    ftp_key = "dnz_vsftp_rsa"
    ftp_user = "vsftp"

    ROOT_PATH = ".."

    GRAND_SERVER_PORT = 10000
    WORKER_SERVER_PORT = 9999

    INS_LIMIT = 2
    TEST_SET_SIZE = 1000

    TRAIN_SET_SIZE = 2000
    MAX_EPOCH = 3
    MAX_ROUNDS = 2

    # TRAIN_SET_SIZE = 20000
    # MAX_EPOCH = 1000
    # MAX_ROUNDS = 100

    # for worker_server
    debug_clear_task_status = False
    debug_remove_all_records = False
    TASK_AGENT_TIMEOUT = 20

    DEVICE = "cuda:0"
    # DEVICE = 'cpu'
