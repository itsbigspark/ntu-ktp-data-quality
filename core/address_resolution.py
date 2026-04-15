import redis
import jellyfish

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def resolve_address(postcode: str, raw_address: str, threshold: float = 0.85):
    postcode = postcode.upper().replace(" ", "")
    key_pc = postcode[:-3] + " " + postcode[-3:]

    # 1) Try exact alias lookup
    alias_key = f"alias:{key_pc}:{raw_address.lower()}"
    direct = r.get(alias_key)
    if direct:
        return direct

    # 2) Retrieve canonical addresses under this postcode
    address_keys = r.smembers(f"postcode:{key_pc}")
    canonical_list = [
        r.hget(f"address:{k}", "canonical_address")
        for k in address_keys
    ]

    # 3) Fuzzy score each candidate
    scored = [
        (addr, jellyfish.jaro_winkler(addr.lower(), raw_address.lower()))
        for addr in canonical_list
    ]

    # Pick the highest scored
    best = max(scored, key=lambda x: x[1])
    if best[1] >= threshold:
        return best[0]

    return None