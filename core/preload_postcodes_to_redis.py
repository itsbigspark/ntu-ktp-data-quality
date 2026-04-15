import pandas as pd
import redis
import json

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Load corpus files
postcodes = pd.read_csv("corpus/postcodes/corpus_postcodes.csv")
addresses = pd.read_csv("corpus/postcodes/corpus_addresses.csv")
aliases   = pd.read_csv("corpus/postcodes/corpus_address_aliases.csv")

print("\n=== LOADING POSTCODE → ADDRESS TREE INTO REDIS ===")

for pc in postcodes['postcode'].unique():
    r.sadd(f"postcode:{pc}", *addresses[addresses.postcode == pc].address_key.values)

print("✔ Postcode roots loaded.")

for _, row in addresses.iterrows():
    data = row.to_dict()
    r.hset(f"address:{row.address_key}", mapping=data)

print("✔ Canonical addresses stored.")

for _, row in aliases.iterrows():
    alias_key = f"alias:{row.postcode}:{row.alias.lower()}"
    r.set(alias_key, row.canonical_address)

print("✔ Alias mappings added.")
print("\n=== REDIS LOAD COMPLETE ===")

