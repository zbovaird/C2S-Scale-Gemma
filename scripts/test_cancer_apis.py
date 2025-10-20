#!/usr/bin/env python3
"""
Test Cancer Data APIs

Test individual APIs to debug issues and verify functionality.
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path

def test_chembl_api():
    """Test ChEMBL API."""
    print("🧪 Testing ChEMBL API...")
    
    try:
        # Test basic API connectivity
        base_url = "https://www.ebi.ac.uk/chembl/api/data"
        
        # Test mechanism endpoint
        url = f"{base_url}/mechanism"
        print(f"Testing URL: {url}")
        
        response = requests.get(url, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ ChEMBL API working! Got {len(data.get('mechanisms', []))} mechanisms")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print(f"Response content: {response.text[:200]}...")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response content: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ ChEMBL API error: {e}")
        return False

def test_string_api():
    """Test STRING API."""
    print("\n🧪 Testing STRING API...")
    
    try:
        # Test STRING API with correct endpoint
        base_url = "https://string-db.org/api"
        
        # Test with a single protein first
        url = f"{base_url}/tsv/network"
        params = {
            'identifiers': 'TP53',
            'species': '9606',  # Human
            'required_score': '400',
            'network_type': 'functional'
        }
        
        print(f"Testing URL: {url}")
        print(f"Params: {params}")
        
        response = requests.get(url, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            print(f"✅ STRING API working! Got {len(lines)} lines")
            if len(lines) > 1:
                print(f"Sample response: {lines[0]}")
                print(f"First data line: {lines[1] if len(lines) > 1 else 'No data'}")
            return True
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response content: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ STRING API error: {e}")
        return False

def test_kegg_api():
    """Test KEGG API."""
    print("\n🧪 Testing KEGG API...")
    
    try:
        # Test KEGG API
        base_url = "https://rest.kegg.jp"
        
        # Test pathway list
        url = f"{base_url}/list/pathway/hsa"
        print(f"Testing URL: {url}")
        
        response = requests.get(url, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            print(f"✅ KEGG API working! Got {len(lines)} pathways")
            if lines:
                print(f"Sample pathway: {lines[0]}")
            return True
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response content: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ KEGG API error: {e}")
        return False

def test_uniprot_api():
    """Test UniProt API."""
    print("\n🧪 Testing UniProt API...")
    
    try:
        # Test UniProt API
        url = "https://www.uniprot.org/uniprot/P04637.xml"  # TP53
        print(f"Testing URL: {url}")
        
        response = requests.get(url, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            content = response.text
            print(f"✅ UniProt API working! Got {len(content)} characters")
            if 'TP53' in content:
                print("✅ TP53 data found in response")
            return True
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ UniProt API error: {e}")
        return False

def test_working_apis():
    """Test APIs that should work."""
    print("\n🧪 Testing working APIs...")
    
    # Test DrugBank (alternative to ChEMBL)
    try:
        print("Testing DrugBank API...")
        # DrugBank has a different API structure
        url = "https://go.drugbank.com/releases/latest"
        response = requests.get(url, timeout=30)
        print(f"DrugBank Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ DrugBank accessible")
        else:
            print("❌ DrugBank not accessible")
    except Exception as e:
        print(f"❌ DrugBank error: {e}")
    
    # Test Reactome (alternative pathway database)
    try:
        print("Testing Reactome API...")
        url = "https://reactome.org/ContentService/data/query/hsa/TP53"
        response = requests.get(url, timeout=30)
        print(f"Reactome Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Reactome accessible")
        else:
            print("❌ Reactome not accessible")
    except Exception as e:
        print(f"❌ Reactome error: {e}")

def main():
    """Test all APIs."""
    print("🔬 Testing Cancer Data APIs")
    print("=" * 50)
    
    results = {}
    
    # Test individual APIs
    results['chembl'] = test_chembl_api()
    results['string'] = test_string_api()
    results['kegg'] = test_kegg_api()
    results['uniprot'] = test_uniprot_api()
    
    # Test alternative APIs
    test_working_apis()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 API Test Results:")
    for api, working in results.items():
        status = "✅ Working" if working else "❌ Not Working"
        print(f"  {api.upper()}: {status}")
    
    working_apis = sum(results.values())
    total_apis = len(results)
    print(f"\n🎯 Summary: {working_apis}/{total_apis} APIs working")
    
    if working_apis >= 2:
        print("✅ Sufficient APIs working for cancer research!")
    else:
        print("⚠️ May need to use alternative data sources")

if __name__ == "__main__":
    main()
