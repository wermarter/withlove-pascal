uses process,fileutil;
var
proc:tprocess;
begin
  if paramstr(1)<>'Hidden' then
  begin
    proc:=tprocess.create(nil);
    proc.options:=[ponoconsole];
    proc.priority:=ppRealTime;
    proc.executable:=paramstrutf8(0);
    proc.parameters.add('Hidden');
    proc.execute;
    proc.free
  end
  else
  begin
    //Your code here
  end
end.