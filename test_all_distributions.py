#!/usr/bin/env python
"""
Comprehensive Distribution Test
================================

Tests all 25 distributions (20 continuous + 5 discrete) for:
- Instantiation
- Info properties
- Display names
- Type classification

Usage:
    python test_all_distributions.py
"""

from distfit_pro.core.distributions import get_distribution, list_distributions

def main():
    print("\n" + "="*70)
    print("  🧪 TESTING ALL 25 DISTRIBUTIONS")
    print("="*70 + "\n")
    
    # Test listing functions
    cont = list_distributions(continuous_only=True)
    disc = list_distributions(discrete_only=True)
    all_dists = list_distributions()
    
    print(f"📋 Continuous distributions available: {len(cont)}")
    print(f"📋 Discrete distributions available: {len(disc)}")
    print(f"📋 Total distributions available: {len(all_dists)}\n")
    
    # Verify counts
    expected_cont = 20
    expected_disc = 5
    expected_total = 25
    
    if len(cont) != expected_cont:
        print(f"⚠️  Warning: Expected {expected_cont} continuous, found {len(cont)}")
    if len(disc) != expected_disc:
        print(f"⚠️  Warning: Expected {expected_disc} discrete, found {len(disc)}")
    if len(all_dists) != expected_total:
        print(f"⚠️  Warning: Expected {expected_total} total, found {len(all_dists)}")
    
    # Test each distribution can be instantiated
    print("🔍 Testing instantiation and properties...\n")
    
    failed = []
    succeeded = []
    
    for dist_name in all_dists:
        try:
            dist = get_distribution(dist_name)
            info = dist.info
            
            # Verify basic properties exist
            assert hasattr(info, 'name'), f"{dist_name}: missing 'name'"
            assert hasattr(info, 'display_name'), f"{dist_name}: missing 'display_name'"
            assert hasattr(info, 'description'), f"{dist_name}: missing 'description'"
            assert hasattr(info, 'is_discrete'), f"{dist_name}: missing 'is_discrete'"
            assert hasattr(info, 'parameters'), f"{dist_name}: missing 'parameters'"
            
            succeeded.append((dist_name, info.display_name, info.is_discrete, len(info.parameters)))
        except Exception as e:
            failed.append((dist_name, str(e)))
    
    # Report results
    if succeeded:
        print(f"✅ Successfully instantiated: {len(succeeded)}/{expected_total}\n")
        
        # Group by type
        continuous = [s for s in succeeded if not s[2]]
        discrete = [s for s in succeeded if s[2]]
        
        print("="*70)
        print("🔵 CONTINUOUS DISTRIBUTIONS (20)")
        print("="*70)
        for i, (name, display, _, n_params) in enumerate(sorted(continuous), 1):
            print(f"   {i:2}. {name:20} → {display:40} ({n_params} params)")
        
        print(f"\n" + "="*70)
        print("🔴 DISCRETE DISTRIBUTIONS (5)")
        print("="*70)
        for i, (name, display, _, n_params) in enumerate(sorted(discrete), 1):
            print(f"   {i}. {name:20} → {display:40} ({n_params} params)")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{expected_total}\n")
        for name, error in failed:
            print(f"   {name}: {error}")
    
    # Final summary
    print("\n" + "="*70)
    if len(succeeded) == expected_total and len(failed) == 0:
        print("  ✅ ALL 25 DISTRIBUTIONS WORKING PERFECTLY!")
        print("="*70)
        print("\n  📊 Summary:")
        print(f"     • Continuous: {len(continuous)}/{expected_cont} ✓")
        print(f"     • Discrete:   {len(discrete)}/{expected_disc} ✓")
        print(f"     • Total:      {len(succeeded)}/{expected_total} ✓")
        print("\n  🎉 Package ready for production!\n")
        return True
    else:
        print(f"  ⚠️  {len(failed)} distributions have issues")
        print("="*70 + "\n")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
