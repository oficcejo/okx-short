import pprint
import json
import time
from okx_client import OKXClient

def debug():
    client = OKXClient()
    inst_id = "ETH-USDT-SWAP"
    
    with open("debug_out.txt", "w", encoding="utf-8") as f:
        f.write("=== Account Balance raw ===\n")
        f.write("\n=== Inst Info ===\n")
        try:
            from okx import PublicData
            pub = PublicData.PublicAPI(flag="0", debug=False)
            res = pub.get_instruments(instType="SWAP", instId="ETH-USDT-SWAP")
            f.write(json.dumps(res, indent=2) + "\n")
        except Exception as e:
            f.write(str(e) + "\n")
        retry = 3
        while retry > 0:
            try:
                res = client.account.get_account_balance(ccy="USDT")
                f.write(json.dumps(res, indent=2) + "\n")
                break
            except Exception as e:
                retry -= 1
                time.sleep(1)
                if retry == 0:
                    f.write(str(e) + "\n")
        
        f.write("\n=== Available Account Methods ===\n")
        f.write(json.dumps([m for m in dir(client.account) if not m.startswith("_")], indent=2))
        
        f.write("\n=== Available Trade Methods ===\n")
        f.write(json.dumps([m for m in dir(client.trade) if not m.startswith("_")], indent=2))
    
if __name__ == "__main__":
    debug()
